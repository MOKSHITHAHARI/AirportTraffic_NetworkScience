"""Protection, rewiring and recovery experiments for the USFN."""
import igraph as ig
import networkx as nx
import numpy as np

import load_cascade as lc


# ------------------------------------------------------------------ attacks with protected airports
def _igraph(g, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    return ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in g.edges])


def _lcc_curve(nodes, edges_idx, order):
    """LCC size after removing order[:k], k = 0..len(order); reverse union-find."""
    n = len(nodes)
    parent = list(range(n)); size = [1] * n; present = [False] * n
    nb = [[] for _ in range(n)]
    for a, b in edges_idx:
        nb[a].append(b); nb[b].append(a)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    best = 0

    def add(i):
        nonlocal best
        present[i] = True; best = max(best, 1)
        for j in nb[i]:
            if present[j]:
                ra, rb = find(i), find(j)
                if ra != rb:
                    if size[ra] < size[rb]: ra, rb = rb, ra
                    parent[rb] = ra; size[ra] += size[rb]; best = max(best, size[ra])
    removed = set(order)
    for i in range(n):
        if i not in removed: add(i)
    out = np.zeros(len(order) + 1, dtype=int); out[-1] = best
    for k in range(len(order) - 1, -1, -1):
        add(order[k]); out[k] = best
    return out


def attack_order(g, protected, kind, nodes=None):
    """Airports (as indices into `nodes`) in the order an attacker removes them. Protected airports are skipped.
    kind: 'degree' / 'betweenness' (ranked once on the intact network) or 'betweenness_adaptive' (re-ranked after every removal)."""
    nodes = nodes or list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    prot = {idx[p] for p in protected}
    h = _igraph(g, nodes)
    if kind in ("degree", "betweenness"):
        sc = np.array(h.degree() if kind == "degree" else h.betweenness(), dtype=float)
        return [i for i in sorted(range(len(nodes)), key=lambda i: (-sc[i], i)) if i not in prot]
    order = []
    hh = h.copy()
    keep = list(range(len(nodes)))                           # mapping from hh vertex ids to original ids
    while True:
        cand = [pos for pos, v in enumerate(keep) if v not in prot]
        if not cand:
            break
        b = np.array(hh.betweenness(), dtype=float)
        top = max(cand, key=lambda pos: (b[pos], -keep[pos]))
        order.append(keep[top])
        hh.delete_vertices(top)
        keep.pop(top)
    return order


def attack_curve(g, protected, kind, nodes=None):
    """LCC size after k removals (k = 0..), for the attack of `attack_order`."""
    nodes = nodes or list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    edges_idx = [(idx[u], idx[v]) for u, v in g.edges]
    return _lcc_curve(nodes, edges_idx, attack_order(g, protected, kind, nodes))


def robustness_R(curve, n):
    return float(np.asarray(curve[1:], dtype=float).mean() / n) if len(curve) > 1 else 1.0


def k_to_half(curve, n):
    hit = np.where(np.asarray(curve) <= 0.5 * n)[0]
    return int(hit[0]) if len(hit) else None


# ------------------------------------------------------------------ cached load-cascade evaluation
class LoadEvaluator:
    """Evaluates protection / capacity configurations for the load cascade on a fixed network. The loads right after each
    origin is removed do not depend on capacities, so they are computed once."""

    def __init__(self, LC):
        self.LC = LC
        n = LC.n
        self.L1 = np.zeros((n, n), dtype=np.float64)
        self.lcc0 = np.zeros(n, dtype=int)
        for s in range(n):
            act = np.ones(n, dtype=np.bool_); act[s] = False
            self.L1[s] = LC.loads(act)
            self.lcc0[s] = LC.lcc_size(act)

    def evaluate(self, cap, protected=(), origins=None):
        """Final largest component for each origin (excluding protected origins); returns array over `origins`."""
        LC = self.LC; n = LC.n
        prot = np.zeros(n, dtype=bool); prot[list(protected)] = True
        cap = cap.copy(); cap[prot] = np.inf
        origins = [s for s in (range(n) if origins is None else origins) if not prot[s]]
        out = np.zeros(len(origins), dtype=int)
        for r, s in enumerate(origins):
            act = np.ones(n, dtype=bool); act[s] = False
            if not (act & (self.L1[s] > cap * (1 + 1e-9) + 1e-9)).any():
                out[r] = self.lcc0[s]
            else:
                out[r] = LC.run(s, cap)["lcc"]
        return np.array(origins), out

    def summary(self, cap, protected=()):
        n = self.LC.n
        origins, fin = self.evaluate(cap, protected)
        return {"dangerous": int((fin <= n / 2).sum()), "share dangerous": float((fin <= n / 2).mean()), "mean LCC / N": float(fin.mean() / n), "min LCC / N": float(fin.min() / n)}


# ------------------------------------------------------------------ rewiring
def great_circle_km(a, b, g):
    la1, lo1, la2, lo2 = map(np.radians, (g.nodes[a]["lat"], g.nodes[a]["lon"], g.nodes[b]["lat"], g.nodes[b]["lon"]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def candidate_pairs(g, max_km, rng, n_sample=60000):
    """Random non-adjacent airport pairs within max_km."""
    nodes = list(g.nodes)
    out, seen = [], set()
    for _ in range(n_sample):
        a, b = rng.choice(len(nodes), 2, replace=False)
        u, v = nodes[a], nodes[b]
        if g.has_edge(u, v) or (u, v) in seen or (v, u) in seen:
            continue
        if great_circle_km(u, v, g) <= max_km:
            out.append((u, v)); seen.add((u, v))
    return out


def lcc_after_removal(g, removed):
    h = g.copy(); h.remove_nodes_from(removed)
    return len(max(nx.connected_components(h), key=len)) if h.number_of_nodes() else 0


def add_edges(g, strategy, m, rng, max_km=2000.0, attack_k=10):
    """Return a copy of g with m new routes chosen by `strategy`:
    random | low_low | dual_home | gateway_bypass | greedy."""
    h = g.copy()
    nodes = list(h.nodes)
    deg = dict(h.degree())
    cand = candidate_pairs(h, max_km, rng)
    chosen = []
    if strategy == "random":
        for i in rng.permutation(len(cand))[:m]:
            chosen.append(cand[i])
    elif strategy == "low_low":
        pool = [(u, v) for u, v in cand if deg[u] <= 5 and deg[v] <= 5]
        for i in rng.permutation(len(pool))[:m]:
            chosen.append(pool[i])
    elif strategy == "dual_home":
        hubs = [n for n in nodes if deg[n] >= 30]
        leaves = [n for n in nodes if deg[n] == 1]
        for i in rng.permutation(len(leaves)):
            if len(chosen) >= m:
                break
            u = leaves[i]
            options = sorted((great_circle_km(u, hb, h), hb) for hb in hubs if hb != u and not h.has_edge(u, hb))
            options = [o for o in options if o[0] <= max_km]
            if options:
                chosen.append((u, options[0][1]))
    elif strategy == "gateway_bypass":
        bc = nx.betweenness_centrality(h)
        tops = sorted(bc, key=bc.get, reverse=True)[:8]
        pool = []
        for v in tops:
            nb = list(h.neighbors(v))
            for a in range(len(nb)):
                for b in range(a + 1, len(nb)):
                    if not h.has_edge(nb[a], nb[b]) and great_circle_km(nb[a], nb[b], h) <= max_km:
                        pool.append((nb[a], nb[b]))
        for i in rng.permutation(len(pool))[:m]:
            chosen.append(pool[i])
    elif strategy == "greedy":
        cur = h.copy()
        for _ in range(m):
            bc = nx.betweenness_centrality(cur)
            attack = sorted(bc, key=bc.get, reverse=True)[:attack_k]
            best, bestv = None, -1
            for i in rng.permutation(len(cand))[:120]:
                u, v = cand[i]
                if cur.has_edge(u, v):
                    continue
                cur.add_edge(u, v)
                val = lcc_after_removal(cur, attack)
                cur.remove_edge(u, v)
                if val > bestv:
                    best, bestv = (u, v), val
            if best:
                cur.add_edge(*best); chosen.append(best)
        return cur, chosen
    for u, v in chosen:
        h.add_edge(u, v)
    return h, chosen


# ------------------------------------------------------------------ recovery
def recovery_curve(g, failed, order):
    """LCC size (relative to the intact network) after restoring `order` airports one by one, starting from the
    damaged network (all `failed` airports absent)."""
    failed = set(failed)
    n = g.number_of_nodes()
    h = g.subgraph([v for v in g if v not in failed]).copy()
    out = [len(max(nx.connected_components(h), key=len)) / n if h.number_of_nodes() else 0.0]
    for v in order:
        h.add_node(v)
        for w in g.neighbors(v):
            if w in h:
                h.add_edge(v, w)
        out.append(len(max(nx.connected_components(h), key=len)) / n)
    return np.array(out)


def greedy_recovery_order(g, failed):
    failed = list(failed)
    n = g.number_of_nodes()
    h = g.subgraph([v for v in g if v not in set(failed)]).copy()
    order, remaining = [], set(failed)
    while remaining:
        best, bestv = None, -1
        for v in sorted(remaining):
            h.add_node(v)
            for w in g.neighbors(v):
                if w in h: h.add_edge(v, w)
            val = len(max(nx.connected_components(h), key=len))
            h.remove_node(v)
            if val > bestv: best, bestv = v, val
        order.append(best); remaining.remove(best)
        h.add_node(best)
        for w in g.neighbors(best):
            if w in h: h.add_edge(best, w)
    return order
