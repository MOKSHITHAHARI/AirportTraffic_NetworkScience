"""Weighted-network tools: two edge weightings of the USFN, Barrat clustering, and service-weighted percolation."""
import networkx as nx
import numpy as np


def _graph_from_rows(rows, coords_from):
    H = nx.Graph()
    for s, t in zip(rows.src, rows.dst):
        if H.has_edge(s, t):
            H[s][t]["weight"] += 1
        else:
            H.add_edge(s, t, weight=1)
    for n in H.nodes:
        if n in coords_from.nodes:
            H.nodes[n].update(coords_from.nodes[n])
    return H


def build_weightings(G, routes):
    """Return (G_all, G_op, codeshare_share).
    G_all = the paper's network (weight = all domestic route rows, codeshare included).
    G_op  = network of pairs with at least one *non-codeshare* row, weight = number of such rows, largest component only.
    codeshare_share[n] = share of airport n's domestic route rows that are codeshare listings."""
    nodes = set(G.nodes)
    dom = routes[routes.src.isin(nodes) & routes.dst.isin(nodes) & (routes.src != routes.dst)]
    op = _graph_from_rows(dom[dom.codeshare != "Y"], G)
    G_op = op.subgraph(max(nx.connected_components(op), key=len)).copy()
    for g in (G, G_op):
        for e in g.edges:
            g.edges[e]["inv"] = 1.0 / g.edges[e]["weight"]
    rows = dom.melt(id_vars=["codeshare"], value_vars=["src", "dst"], value_name="airport")
    share = rows.groupby("airport").codeshare.apply(lambda s: float((s == "Y").mean()))
    return G, G_op, share.to_dict()


def barrat_clustering(G, weight="weight"):
    """Barrat et al. (2004) weighted local clustering; equals the unweighted coefficient if all weights are equal."""
    out = {}
    for i in G.nodes:
        k = G.degree(i)
        if k < 2:
            out[i] = 0.0
            continue
        s = G.degree(i, weight=weight)
        nb = list(G.neighbors(i))
        tot = 0.0
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                if G.has_edge(nb[a], nb[b]):
                    tot += G[i][nb[a]][weight] + G[i][nb[b]][weight]          # (w_ij + w_ih)/2 counted for both orderings
        out[i] = tot / (s * (k - 1))
    return out


def service_percolation(G, order, weight="weight"):
    """For k = 0..len(order): after removing order[:k], return
    lcc_size, lcc_weight (total edge weight inside the largest component) and surviving_weight (weight among all survivors).
    One reverse pass with union-find that tracks the weight inside each component."""
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    nb = [[(idx[m], G[n][m][weight]) for m in G.neighbors(n)] for n in nodes]
    parent, size, cw = list(range(N)), [1] * N, [0.0] * N
    present = [False] * N
    st = {"surv": 0.0, "best": -1}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def add(i):
        present[i] = True
        if st["best"] < 0:
            st["best"] = i
        for j, w in nb[i]:
            if present[j]:
                st["surv"] += w
                ra, rb = find(i), find(j)
                if ra == rb:
                    cw[ra] += w
                else:
                    if size[ra] < size[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    size[ra] += size[rb]
                    cw[ra] += cw[rb] + w
                rr = find(i)
                if size[rr] > size[find(st["best"])]:
                    st["best"] = rr

    removed = {idx[n] for n in order}
    for i in range(N):
        if i not in removed:
            add(i)
    K = len(order)
    lcc, lw, sw = np.zeros(K + 1, dtype=int), np.zeros(K + 1), np.zeros(K + 1)

    def rec(k):
        r = find(st["best"]) if st["best"] >= 0 else None
        lcc[k] = size[r] if r is not None else 0
        lw[k] = cw[r] if r is not None else 0.0
        sw[k] = st["surv"]

    rec(K)
    for k in range(K - 1, -1, -1):
        add(idx[order[k]])
        rec(k)
    return lcc, lw, sw
