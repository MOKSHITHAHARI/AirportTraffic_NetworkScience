"""Reproduction of 'Analysis and Resilience of the U.S. Flight Network' (arXiv:2505.11559)."""
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

AIRPORT_COLS = ["id", "name", "city", "country", "iata", "icao", "lat", "lon",
                "alt", "tz", "dst", "tzdb", "type", "source"]
ROUTE_COLS = ["airline", "airline_id", "src", "src_id", "dst", "dst_id",
              "codeshare", "stops", "equip"]


def load_openflights():
    airports = pd.read_csv(RAW / "airports.dat", header=None, names=AIRPORT_COLS, na_values=["\\N"])
    routes = pd.read_csv(RAW / "routes.dat", header=None, names=ROUTE_COLS, na_values=["\\N"])
    return airports, routes


def build_country_network(airports, routes, country="United States", largest_component=True):
    """Weighted undirected network; weight = number of route records between the two airports."""
    sub = airports[airports.country == country].dropna(subset=["iata"])
    codes = set(sub.iata)
    dom = routes[routes.src.isin(codes) & routes.dst.isin(codes)]
    G = nx.Graph()
    for s, t in zip(dom.src, dom.dst):
        if s == t:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += 1
        else:
            G.add_edge(s, t, weight=1)
    info = sub.drop_duplicates("iata").set_index("iata")
    for n in G.nodes:
        r = info.loc[n]
        G.nodes[n].update(name=str(r["name"]), city=str(r["city"]), lat=float(r["lat"]), lon=float(r["lon"]))
    if largest_component:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def basic_stats(G):
    deg = dict(G.degree())
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
        "max_degree": max(deg.items(), key=lambda x: x[1]),
        "avg_clustering": nx.average_clustering(G),
        "avg_shortest_path": nx.average_shortest_path_length(G),
        "diameter": nx.diameter(G),
    }


def powerlaw_exponent_loglog(G):
    """Least-squares slope of the log-log degree distribution (what the paper's Fig. 3 does)."""
    degs = np.array([d for _, d in G.degree()])
    ks, counts = np.unique(degs, return_counts=True)
    pk = counts / counts.sum()
    slope, _ = np.polyfit(np.log(ks), np.log(pk), 1)
    return -slope


def powerlaw_exponent_mle(G, kmin=1):
    """Clauset-style continuous MLE approximation, for comparison with the log-log fit."""
    degs = np.array([d for _, d in G.degree() if d >= kmin])
    return 1 + len(degs) / np.sum(np.log(degs / (kmin - 0.5)))


def powerlaw_fit_report(G):
    """Clauset-Shalizi-Newman fit plus likelihood-ratio tests against alternatives."""
    import powerlaw
    degs = np.array([d for _, d in G.degree()])
    fit = powerlaw.Fit(degs, discrete=True, verbose=False)
    out = {"alpha": fit.power_law.alpha, "xmin": fit.power_law.xmin}
    for alt in ("lognormal", "exponential", "truncated_power_law"):
        R, p = fit.distribution_compare("power_law", alt)
        out[f"R_vs_{alt}"] = R
        out[f"p_vs_{alt}"] = p
    return out


def louvain_vs_labelprop(G, seed=42):
    lou = nx.community.louvain_communities(G, seed=seed)
    lp = list(nx.community.asyn_lpa_communities(G, seed=seed))
    return {
        "louvain": (lou, nx.community.modularity(G, lou)),
        "label_propagation": (lp, nx.community.modularity(G, lp)),
    }


def community_table(G, comms):
    rows = []
    for i, c in enumerate(sorted(comms, key=len, reverse=True)):
        sg = G.subgraph(c)
        rows.append({"community": i, "size": len(c), "edges": sg.number_of_edges(), "density": nx.density(sg)})
    return pd.DataFrame(rows)


def null_models(G, n_each=3, seed=0):
    N, m = G.number_of_nodes(), G.number_of_edges()
    p = 2 * m / (N * (N - 1))
    out = []
    for i in range(n_each):
        er = nx.gnp_random_graph(N, p, seed=seed + i)
        if not nx.is_connected(er):
            er = er.subgraph(max(nx.connected_components(er), key=len)).copy()
        out.append(("ER", er))
    k = max(1, round(m / N))
    for i in range(n_each):
        out.append(("BA", nx.barabasi_albert_graph(N, k, seed=seed + i)))
    rows = [{"model": "Real", "nodes": N, "edges": m, "C": nx.average_clustering(G),
             "L": nx.average_shortest_path_length(G)}]
    for name, g in out:
        rows.append({"model": name, "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                     "C": nx.average_clustering(g), "L": nx.average_shortest_path_length(g)})
    return pd.DataFrame(rows)


def lcc_size(G):
    return len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() else 0


def removal_curve(G, order, k):
    H = G.copy()
    sizes = [lcc_size(H)]
    for n in order[:k]:
        H.remove_node(n)
        sizes.append(lcc_size(H))
    return sizes


def attack_curves(G, k=30, seed=0, n_random=20):
    """Static-ranking attacks (paper's approach) plus random-failure average."""
    rng = random.Random(seed)
    rank = lambda sc: sorted(sc, key=sc.get, reverse=True)
    orders = {
        "degree": rank(dict(G.degree())),
        "betweenness": rank(nx.betweenness_centrality(G)),
        "closeness": rank(nx.closeness_centrality(G)),
    }
    curves = {name: removal_curve(G, o, k) for name, o in orders.items()}
    nodes = list(G.nodes)
    rand = np.mean([removal_curve(G, rng.sample(nodes, k), k) for _ in range(n_random)], axis=0)
    curves["random"] = list(rand)
    return curves, orders


def threshold_cascade(G, seed_node, threshold):
    """Fractional-threshold cascade: a node fails once failed_neighbors/degree >= threshold."""
    failed = {seed_node}
    frontier = {seed_node}
    steps = 0
    while frontier:
        cand = set()
        for f in frontier:
            cand.update(n for n in G.neighbors(f) if n not in failed)
        new = {n for n in cand
               if sum(1 for x in G.neighbors(n) if x in failed) / G.degree(n) >= threshold}
        if not new:
            break
        failed |= new
        frontier = new
        steps += 1
    return failed, steps


def threshold_sweep(G, seed_node=None, thresholds=None):
    seed_node = seed_node or max(G.degree, key=lambda x: x[1])[0]
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.05, 0.501, 0.01), 2)
    rows = []
    for t in thresholds:
        failed, steps = threshold_cascade(G, seed_node, t)
        rows.append({"threshold": t, "failed": len(failed), "fraction": len(failed) / G.number_of_nodes(), "steps": steps})
    return pd.DataFrame(rows)
