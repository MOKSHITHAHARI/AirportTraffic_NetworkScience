"""Power-law fitting for integer data (degrees), after Clauset, Shalizi & Newman (2009)."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import zeta

ALPHA_BOUNDS = (1.01, 6.0)


def _candidates(x, min_tail):
    ux = np.unique(x)
    return [u for u in ux if (x >= u).sum() >= min_tail]


def _ks_discrete(tail, xmin, alpha):
    ux, c = np.unique(tail, return_counts=True)
    emp = np.cumsum(c) / c.sum()
    model = 1.0 - zeta(alpha, ux + 1) / zeta(alpha, xmin)
    return float(np.max(np.abs(emp - model)))


def _ks_continuous(tail, xmin, alpha):
    """KS distance in the `powerlaw` package's convention for tied (integer) data: the empirical CDF
    just before each distinct value is compared with the continuous model CDF at that value."""
    ux, c = np.unique(tail, return_counts=True)
    emp_before = np.concatenate([[0.0], np.cumsum(c)[:-1]]) / c.sum()
    model = 1.0 - (ux / xmin) ** (1.0 - alpha)
    return float(np.max(np.abs(emp_before - model)))


def alpha_discrete(x, xmin):
    """Exact discrete power-law MLE for a fixed xmin (maximises the Hurwitz-zeta likelihood)."""
    tail = x[x >= xmin].astype(float)
    n, slog = len(tail), np.log(tail).sum()
    res = minimize_scalar(lambda a: a * slog + n * np.log(zeta(a, xmin)), bounds=ALPHA_BOUNDS, method="bounded")
    return float(res.x)


def alpha_continuous(x, xmin):
    """Continuous power-law MLE (closed form); what the `powerlaw` package does by default."""
    tail = x[x >= xmin].astype(float)
    return float(1.0 + len(tail) / np.log(tail / xmin).sum())


def fit(x, kind="discrete", xmin=None, min_tail=10):
    """Fit alpha; if xmin is None choose it by minimising the KS distance. Returns dict."""
    x = np.asarray(x)
    est, ks = (alpha_discrete, _ks_discrete) if kind == "discrete" else (alpha_continuous, _ks_continuous)
    cands = [xmin] if xmin is not None else _candidates(x, min_tail)
    best = None
    for xm in cands:
        a = est(x, xm)
        d = ks(x[x >= xm], xm, a)
        if best is None or d < best["ks"]:
            best = {"alpha": a, "xmin": float(xm), "ks": d, "n_tail": int((x >= xm).sum())}
    return best


def sample_discrete(n, alpha, xmin, rng, cap=10**6):
    """Draw n values from a discrete power law on {xmin..cap} by inverse-CDF sampling."""
    k = np.arange(int(xmin), cap + 1, dtype=float)
    cdf = np.cumsum(k ** (-alpha))
    cdf /= cdf[-1]
    return (np.searchsorted(cdf, rng.random(n)) + int(xmin)).astype(int)


def gof_bootstrap(x, kind, n_boot, seed=0, min_tail=10):
    """Semi-parametric goodness-of-fit p-value (CSN section 4.1); small p rejects the power law."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    f = fit(x, kind, min_tail=min_tail)
    xmin, alpha, n = int(f["xmin"]), f["alpha"], len(x)
    body = x[x < xmin]
    p_tail = f["n_tail"] / n
    ks_null = np.empty(n_boot)
    for b in range(n_boot):
        is_tail = rng.random(n) < p_tail
        synth = np.empty(n, dtype=int)
        synth[is_tail] = sample_discrete(int(is_tail.sum()), alpha, xmin, rng)
        synth[~is_tail] = rng.choice(body, size=int((~is_tail).sum())) if len(body) else xmin
        ks_null[b] = fit(synth, kind, min_tail=min_tail)["ks"]
    return {"fit": f, "ks_null": ks_null, "p": float((ks_null >= f["ks"]).mean())}


def bootstrap_alpha(x, kind, n_boot, seed=0, min_tail=10):
    """Non-parametric bootstrap of (alpha, xmin): resample the degrees with replacement and refit."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    out = np.empty((n_boot, 2))
    for b in range(n_boot):
        r = fit(rng.choice(x, size=len(x)), kind, min_tail=min_tail)
        out[b] = (r["alpha"], r["xmin"])
    return out


# ---------------------------------------------------------------------------------------------
# Competing discrete models on the tail k >= xmin, with likelihood-ratio tests against the power law
# ---------------------------------------------------------------------------------------------
from scipy.optimize import minimize
from scipy.special import erfc
from scipy.stats import chi2, norm

KMAX = 200_000


def _pl_loglik_i(t, xmin, alpha):
    return -alpha * np.log(t) - np.log(zeta(alpha, xmin))


def fit_alternatives(x, xmin):
    """Fit exponential, log-normal and power-law-with-cutoff to the tail (all discrete) and return
    per-observation log-likelihoods and parameters, plus the pure power law for reference."""
    t = x[x >= xmin].astype(float)
    n = len(t)
    k = np.arange(int(xmin), KMAX + 1, dtype=float)
    out = {}

    a_pl = alpha_discrete(x, xmin)
    out["power law"] = {"params": {"alpha": a_pl}, "ll_i": _pl_loglik_i(t, xmin, a_pl), "k": 1}

    m = np.mean(t - xmin)
    lam = np.log(1.0 + 1.0 / m)
    out["exponential"] = {"params": {"lambda": lam}, "ll_i": np.log1p(-np.exp(-lam)) - lam * (t - xmin), "k": 1}

    def ln_pmf(v, mu, sg):
        d = norm.cdf((np.log(v + 0.5) - mu) / sg) - norm.cdf((np.log(v - 0.5) - mu) / sg)
        return np.log(np.maximum(d, 1e-300))

    def ln_logZ(mu, sg):
        return np.log(max(1.0 - norm.cdf((np.log(xmin - 0.5) - mu) / sg), 1e-300))

    def ln_nll(p):
        mu, sg = p[0], np.exp(p[1])
        return -(ln_pmf(t, mu, sg).sum() - n * ln_logZ(mu, sg))

    r = minimize(ln_nll, [np.log(t).mean(), np.log(np.log(t).std() + 1e-6)], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000})
    mu, sg = r.x[0], float(np.exp(r.x[1]))
    out["log-normal"] = {"params": {"mu": mu, "sigma": sg}, "ll_i": ln_pmf(t, mu, sg) - ln_logZ(mu, sg), "k": 2}

    lk = np.log(k)

    def cut_nll(p):
        a, lm = p
        return -(np.sum(-a * np.log(t) - lm * t) - n * np.log(np.sum(np.exp(-a * lk - lm * k))))

    r = minimize(cut_nll, [a_pl, 1e-3], method="L-BFGS-B", bounds=[(0.3, 6.0), (1e-9, 1.0)])
    a_c, lm = float(r.x[0]), float(r.x[1])
    logZ = np.log(np.sum(np.exp(-a_c * lk - lm * k)))
    out["power law + cutoff"] = {"params": {"alpha": a_c, "lambda": lm}, "ll_i": -a_c * np.log(t) - lm * t - logZ, "k": 2}
    return out


def compare_to_power_law(models):
    """Vuong test (non-nested) / chi-square boundary test (nested cutoff) of each alternative vs the power law.
    R > 0 favours the power law, R < 0 the alternative."""
    base = models["power law"]["ll_i"]
    n = len(base)
    rows = []
    for name, m in models.items():
        ll_total = m["ll_i"].sum()
        row = {"model": name, "n_params": m["k"], "log-likelihood": ll_total, "AIC": 2 * m["k"] - 2 * ll_total}
        if name != "power law":
            d = base - m["ll_i"]
            R = d.sum()
            if name == "power law + cutoff":                       # nested: lambda = 0 recovers the power law
                stat = max(2 * (ll_total - base.sum()), 0.0)
                p = 0.5 * chi2.sf(stat, 1)
            else:
                z = R / (np.sqrt(n) * d.std(ddof=0))
                p = float(erfc(abs(z) / np.sqrt(2)))
            row.update({"R (PL vs model)": R, "p-value": p})
        rows.append(row)
    return rows


def model_ccdf(model_name, params, xmin, kgrid):
    """P(K >= k | K >= xmin) for each fitted discrete model, evaluated on integer grid kgrid."""
    k = np.arange(int(xmin), KMAX + 1, dtype=float)
    if model_name == "power law":
        pmf = k ** (-params["alpha"])
    elif model_name == "exponential":
        pmf = np.exp(-params["lambda"] * (k - xmin))
    elif model_name == "log-normal":
        pmf = norm.cdf((np.log(k + 0.5) - params["mu"]) / params["sigma"]) - norm.cdf((np.log(k - 0.5) - params["mu"]) / params["sigma"])
    else:
        pmf = k ** (-params["alpha"]) * np.exp(-params["lambda"] * k)
    pmf = pmf / pmf.sum()
    surv = 1.0 - np.concatenate([[0.0], np.cumsum(pmf)[:-1]])          # P(K >= k)
    return surv[(np.asarray(kgrid) - int(xmin)).astype(int)]
