"""Build networks for countries / regions / the world from OpenFlights and profile them with the project's measures."""
import igraph as ig
import networkx as nx
import numpy as np

import attacks as at
import cascade as cs
import communities as cm
import load_cascade as lc
import nullmodels as nm
import powerlaw_tools as pt

EUROPE = ["Albania", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
          "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Macedonia", "Malta",
          "Moldova", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
          "Switzerland", "Ukraine", "United Kingdom"]


def _lcc(H):
    if H.number_of_nodes() == 0:
        return H
    return H.subgraph(max(nx.connected_components(H), key=len)).copy()


def build_region_network(airports, routes, countries=None):
    """Unweighted network of airports whose country is in `countries` (None = every airport), joined by any route
    between two of them; largest connected component only. Node attributes: name, city, lat, lon, country."""
    ap = airports.dropna(subset=["iata"]).drop_duplicates("iata").set_index("iata")
    if countries is not None:
        ap = ap[ap.country.isin(countries)]
    keep = set(ap.index)
    H = nx.Graph()
    for s, t in zip(routes.src, routes.dst):
        if s != t and s in keep and t in keep:
            H.add_edge(s, t)
    H = _lcc(H)
    for n in H.nodes:
        r = ap.loc[n]
        H.nodes[n].update(name=str(r["name"]), city=str(r["city"]), lat=float(r["lat"]), lon=float(r["lon"]), country=str(r["country"]))
    return H


def _ig(g):
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    return nodes, ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in g.edges])


def fast_order(g, kind):
    """Static attack order using igraph (fast on large networks); ties broken by node position."""
    nodes, h = _ig(g)
    sc = {"degree": h.degree(), "betweenness": h.betweenness(), "closeness": h.closeness()}[kind]
    sc = np.nan_to_num(np.array(sc, dtype=float))
    return [nodes[i] for i in sorted(range(len(nodes)), key=lambda i: (-sc[i], i))]


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((n + 1 - 2 * (np.cumsum(x).sum() / x.sum())) / n)


def profile(g, seed=0, n_boot=200, n_null=20, n_random=60, thresholds=True, loadcascade=True, all_origins=True):
    """Structure, heavy tail, communities, attacks, threshold cascade, and load cascade of one network."""
    rng = np.random.default_rng(seed)
    nodes = list(g.nodes)
    n, m = len(nodes), g.number_of_edges()
    dg = dict(g.degree())
    deg = np.array([dg[v] for v in nodes])
    hub = max(dg, key=dg.get)
    out = {"airports": n, "edges": m, "mean degree": 2 * m / n, "largest hub": hub, "largest degree": int(deg.max()), "hub dominance (kmax/(N-1))": deg.max() / (n - 1),
           "share of degree-1 airports": float((deg == 1).mean()), "degree Gini": gini(deg)}
    met = nm.fast_metrics(g)
    out.update({"clustering C": met["C"], "path length L": met["L"], "diameter": met["diameter"], "assortativity": met["assortativity"]})
    # heavy tail
    f = pt.fit(deg, "discrete")
    out["gamma (discrete MLE)"] = f["alpha"]
    out["gamma x_min"] = int(f["xmin"])
    out["power-law GOF p"] = pt.gof_bootstrap(deg, "discrete", n_boot=n_boot, seed=seed)["p"] if n_boot else np.nan
    # communities against a degree-preserving null
    Gu = nx.Graph(g.edges)
    q = [nx.community.modularity(Gu, cm.louvain(Gu, s, weight=None), weight=None) for s in range(8)]
    out["Louvain modularity"] = float(np.mean(q))
    if n_null:
        qn = [nx.community.modularity(nm.degree_preserving(Gu, 3000 + s), cm.louvain(nm.degree_preserving(Gu, 3000 + s), 0, weight=None), weight=None) for s in range(n_null)]
        out["modularity z vs degree-preserving null"] = (np.mean(q) - np.mean(qn)) / (np.std(qn, ddof=1) + 1e-12)
    # attacks
    curves = {}
    for kind in ("degree", "betweenness"):
        curves[kind] = at.percolation_curves(g, fast_order(g, kind))[0]
    rl = np.mean([at.percolation_curves(g, list(rng.permutation(nodes)))[0] for _ in range(n_random)], axis=0)
    out["random: removals to halve LCC / N"] = at.k_to_reach(rl, n, .5) / n
    for kind in ("degree", "betweenness"):
        out[f"{kind} attack: removals to halve LCC / N"] = at.k_to_reach(curves[kind], n, .5) / n
        out[f"{kind} attack: robustness R"] = at.robustness_R(curves[kind], n)
    out["random: robustness R"] = at.robustness_R(rl, n)
    k5 = max(1, round(0.05 * n))
    out["LCC share after removing top 5% by betweenness"] = curves["betweenness"][k5] / n
    out["LCC / survivors after removing the top hub (betweenness)"] = curves["betweenness"][1] / (n - 1)
    out["LCC / survivors after removing the top 3 (betweenness)"] = curves["betweenness"][3] / (n - 3)
    # threshold cascade (M6)
    if thresholds:
        C = cs.Cascade(g)
        tg = np.round(np.arange(0.02, 0.4001, 0.005), 3)
        hubi = int(np.argmax(C.deg))
        ok = [t for t in tg if C.run([hubi], C.need(t))[0].sum() >= 0.5 * n]
        out["threshold cascade: critical threshold of the top hub"] = max(ok) if ok else 0.0
        if all_origins:
            tcs = []
            for i in range(n):
                okk = [t for t in tg if C.run([i], C.need(t))[0].sum() >= 0.5 * n]
                tcs.append(max(okk) if okk else 0.0)
            tcs = np.array(tcs)
            out["threshold cascade: worst-origin critical threshold"] = float(tcs.max())
            out["threshold cascade: worst origin"] = C.nodes[int(tcs.argmax())]
            out["threshold cascade: share of origins causing an avalanche at t = 0.10"] = float(np.mean([C.run([i], C.need(0.10))[0].sum() >= 0.5 * n for i in range(n)]))
    # load cascade (M9)
    if loadcascade:
        LC = lc.LoadCascade(g)
        al = np.round(np.arange(0, 1.0001, 0.05), 3)
        R, _ = LC.sweep(al)
        gl = R <= n / 2
        ac = np.array([al[np.where(gl[i])[0]].max() if gl[i].any() else -1.0 for i in range(n)])
        out["load cascade: critical spare capacity alpha*"] = float(ac.max())
        out["load cascade: share of origins causing an avalanche at alpha = 0"] = float(gl[:, 0].mean())
        out["load cascade: share at alpha = 0.1"] = float(gl[:, int(np.argmin(abs(al - 0.1)))].mean())
        out["load cascade: most dangerous origin"] = LC.nodes[int(ac.argmax())] if ac.max() >= 0 else "none"
    return out
