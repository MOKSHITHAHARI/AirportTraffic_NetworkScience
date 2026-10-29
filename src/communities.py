"""Community-detection helpers for the USFN."""
import random

import networkx as nx
import numpy as np
import pandas as pd


def louvain(G, seed, weight="weight"):
    return [set(c) for c in nx.community.louvain_communities(G, weight=weight, seed=seed)]


def label_propagation_shuffled(G, seed):
    """networkx's (semi-synchronous) label propagation, run on a copy whose nodes are inserted in a random order.
    The algorithm is deterministic given the node order, so this is how its run-to-run randomness is explored."""
    nodes = list(G.nodes)
    random.Random(seed).shuffle(nodes)
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **d)
    return [set(c) for c in nx.community.label_propagation_communities(H)]


def async_label_propagation(G, seed, weight="weight"):
    return [set(c) for c in nx.community.asyn_lpa_communities(G, weight=weight, seed=seed)]


def greedy_modularity(G, weight="weight"):
    return [set(c) for c in nx.community.greedy_modularity_communities(G, weight=weight)]


def leiden(G, seed, weight="weight"):
    import igraph as ig
    import leidenalg
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges])
    w = [G[u][v][weight] for u, v in G.edges] if weight else None
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=w, seed=seed)
    return [{nodes[i] for i in c} for c in part]


def infomap(G, seed, weight="weight"):
    """Infomap via python-igraph (flow-based communities, not modularity-based)."""
    import igraph as ig
    import random as _r
    _r.seed(seed)
    ig.set_random_number_generator(_r)
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges])
    w = [G[u][v][weight] for u, v in G.edges] if weight else None
    part = g.community_infomap(edge_weights=w, trials=5)
    return [{nodes[i] for i in c} for c in part]


def modularity_both(G, comms):
    return nx.community.modularity(G, comms), nx.community.modularity(G, comms, weight=None)


def partition_table(G, comms):
    rows = []
    for i, c in enumerate(sorted(comms, key=len, reverse=True)):
        sg = G.subgraph(c)
        rows.append({"community": i, "size": len(c), "edges": sg.number_of_edges(), "density": nx.density(sg)})
    return pd.DataFrame(rows)


def spread(tbl):
    """Population standard deviations of community size, edge count and density (numpy default), as in the paper's Table 2."""
    return {"size_std": tbl["size"].std(ddof=0), "edges_std": tbl["edges"].std(ddof=0), "density_std": tbl["density"].std(ddof=0)}


def labels(comms, nodes):
    lab = {n: i for i, c in enumerate(comms) for n in c}
    return np.array([lab[n] for n in nodes])
