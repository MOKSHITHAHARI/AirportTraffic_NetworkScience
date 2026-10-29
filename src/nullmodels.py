"""Null models for the USFN: Erdos-Renyi, Barabasi-Albert, degree-preserving rewiring, plus fast metrics."""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def fast_metrics(g):
    """Structural metrics; path statistics use only reachable pairs, so they also work on disconnected graphs."""
    A = csr_matrix(nx.to_scipy_sparse_array(g, format="csr", dtype=float))
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    D = shortest_path(A, method="D", unweighted=True, directed=False)
    reach = np.isfinite(D) & (D > 0)
    deg = np.array([d for _, d in g.degree()])
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "C": nx.average_clustering(g),
        "transitivity": nx.transitivity(g),
        "L": D[reach].sum() / reach.sum(),
        "diameter": int(D[np.isfinite(D)].max()),
        "assortativity": nx.degree_assortativity_coefficient(g),
        "max_degree": int(deg.max()),
        "share_degree1": float((deg == 1).mean()),
        "connected": nx.is_connected(g),
    }


def er_like(n, m, seed):
    """Erdos-Renyi graph with p = 2m / (n(n-1)), exactly as in the paper."""
    return nx.gnp_random_graph(n, 2 * m / (n * (n - 1)), seed=seed)


def ba_like(n, m, seed):
    """Barabasi-Albert graph; each new node attaches with k = round(m/n) edges (k = 5 for the USFN)."""
    return nx.barabasi_albert_graph(n, max(1, round(m / n)), seed=seed)


def degree_preserving(G, seed, swaps_per_edge=10):
    """Maslov-Sneppen rewiring: swap edge endpoints repeatedly, keeping every node's degree fixed."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from(G.edges)
    m = H.number_of_edges()
    nx.double_edge_swap(H, nswap=swaps_per_edge * m, max_tries=1000 * m, seed=seed)
    return H


def great_circle_km(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def edge_lengths_km(g, G):
    """Great-circle length of every edge of g, using airport coordinates stored on the real graph G."""
    us, vs = zip(*g.edges)
    lat = lambda ns: np.array([G.nodes[n]["lat"] for n in ns])
    lon = lambda ns: np.array([G.nodes[n]["lon"] for n in ns])
    return great_circle_km(lat(us), lon(us), lat(vs), lon(vs))


def all_pairs_mean_km(G):
    """Mean distance between two randomly chosen airports: the edge length of a geography-blind random graph."""
    nodes = list(G.nodes)
    lat = np.array([G.nodes[n]["lat"] for n in nodes])
    lon = np.array([G.nodes[n]["lon"] for n in nodes])
    i, j = np.triu_indices(len(nodes), k=1)
    return float(great_circle_km(lat[i], lon[i], lat[j], lon[j]).mean())
