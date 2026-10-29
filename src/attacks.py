"""Node-removal (attack) experiments: orderings, percolation curves, and damage measures."""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


# ------------------------------------------------------------------ orderings
def _rank(scores, nodes):
    """Descending by score; ties broken by position in `nodes` (stable)."""
    pos = {n: i for i, n in enumerate(nodes)}
    return sorted(scores, key=lambda n: (-scores[n], pos[n]))


def scorer(name):
    """Centrality scorers on an (unweighted) graph; each returns {node: score}."""
    if name == "degree":
        return lambda g: dict(g.degree())
    if name == "strength":
        return lambda g: dict(g.degree(weight="weight"))
    if name == "betweenness":
        return lambda g: nx.betweenness_centrality(g)
    if name == "closeness":
        return lambda g: nx.closeness_centrality(g)
    if name == "pagerank":
        return lambda g: nx.pagerank(g, weight=None)
    if name == "eigenvector":
        return lambda g: nx.eigenvector_centrality_numpy(g)
    if name == "kcore":
        def f(g):
            core = nx.core_number(g)
            return {n: core[n] + g.degree(n) * 1e-6 for n in g}       # coreness, ties broken by degree
        return f
    raise ValueError(name)


def static_order(G, name):
    """Rank once on the intact network and remove in that order (the paper's approach)."""
    return _rank(scorer(name)(G), list(G.nodes))


def adaptive_order(G, name, n_steps=None):
    """Recompute the centrality on the remaining network after every removal and remove the current top node."""
    nodes = list(G.nodes)
    H = G.copy()
    f = scorer(name)
    order = []
    for _ in range(n_steps or len(nodes)):
        if H.number_of_nodes() == 0:
            break
        sc = f(H)
        top = _rank(sc, [n for n in nodes if n in H])[0]
        order.append(top)
        H.remove_node(top)
    return order


# ------------------------------------------------------------------ percolation curves
def percolation_curves(G, order):
    """For k = 0..len(order): size of the largest component, fraction of connected (ordered) node pairs, and number of
    components after removing order[:k]. Computed in one pass by adding the nodes back in reverse order (union-find)."""
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    nbrs = [[idx[m] for m in G.neighbors(n)] for n in nodes]
    parent = list(range(N))
    size = [1] * N
    present = [False] * N
    st = {"lcc": 0, "s2": 0, "comp": 0}                           # s2 = sum over components of s(s-1) = connected ordered pairs

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def add(i):
        present[i] = True
        st["comp"] += 1
        st["lcc"] = max(st["lcc"], 1)
        for j in nbrs[i]:
            if present[j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    sa, sb = size[ra], size[rb]
                    st["s2"] += (sa + sb) * (sa + sb - 1) - sa * (sa - 1) - sb * (sb - 1)
                    if sa < sb:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    size[ra] = sa + sb
                    st["comp"] -= 1
                    st["lcc"] = max(st["lcc"], size[ra])

    removed = {idx[n] for n in order}
    for i in range(N):
        if i not in removed:
            add(i)
    K = len(order)
    lcc, pair, comp = np.zeros(K + 1, dtype=int), np.zeros(K + 1), np.zeros(K + 1, dtype=int)
    lcc[K], pair[K], comp[K] = st["lcc"], st["s2"] / (N * (N - 1)), st["comp"]
    for k in range(K - 1, -1, -1):
        add(idx[order[k]])
        lcc[k], pair[k], comp[k] = st["lcc"], st["s2"] / (N * (N - 1)), st["comp"]
    return lcc, pair, comp


def efficiency_at(G, order, ks):
    """Global efficiency after removing order[:k], for each k in ks: sum of 1/d over ordered pairs of remaining
    nodes divided by the intact network's value over all N(N-1) pairs (removed nodes contribute 0)."""
    nodes = list(G.nodes)
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    A = csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float))

    def eff(keep):
        sub = A[keep][:, keep]
        sub.indices = sub.indices.astype(np.int32)
        sub.indptr = sub.indptr.astype(np.int32)
        D = shortest_path(sub, method="D", unweighted=True, directed=False)
        with np.errstate(divide="ignore"):
            inv = np.where(np.isfinite(D) & (D > 0), 1.0 / D, 0.0)
        return inv.sum()

    base = eff(np.arange(N))
    out = []
    for k in ks:
        removed = np.zeros(N, dtype=bool)
        removed[[idx[n] for n in order[:k]]] = True
        out.append(eff(np.where(~removed)[0]) / base if (~removed).sum() > 1 else 0.0)
    return np.array(out)


# ------------------------------------------------------------------ summary measures
def robustness_R(lcc, N):
    """Schneider et al. robustness index: mean LCC fraction over k = 1..N removals (1 = perfectly robust, ~0 = fragile)."""
    lcc = np.asarray(lcc, dtype=float)
    return float(lcc[1:N + 1].mean() / N)


def k_to_reach(lcc, N, frac):
    """Smallest number of removals after which the largest component is at most frac * N (None if never)."""
    hit = np.where(np.asarray(lcc) <= frac * N)[0]
    return int(hit[0]) if len(hit) else None
