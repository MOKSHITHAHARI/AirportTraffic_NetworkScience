"""Fractional-threshold (Watts-style) cascade on a network, with integer arithmetic to avoid float ties."""
import numpy as np


class Cascade:
    """Precomputed adjacency for fast repeated cascades on a fixed graph G."""

    def __init__(self, G):
        self.nodes = list(G.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.nbrs = [np.array([self.idx[m] for m in G.neighbors(n)], dtype=np.int64) for n in self.nodes]
        self.deg = np.array([len(a) for a in self.nbrs])

    def need(self, t, rule=">="):
        """Minimum number of failed neighbours that makes each node fail at threshold t.
        rule '>=' : fails when failed/degree >= t (the paper's Fig. 14 numbers need this);  '>' : strictly greater."""
        d = self.deg.astype(float)
        if rule == ">=":
            k = np.ceil(t * d - 1e-9)
        else:
            k = np.floor(t * d + 1e-9) + 1
        return np.maximum(k, 1).astype(np.int64)

    def run(self, seeds, need):
        """Synchronous rounds. `seeds` are node names or indices. Returns (failed bool array, step array, rounds).
        step[i] = round in which node i failed (0 for the seeds), -1 if it never failed."""
        seeds = [self.idx[s] if not isinstance(s, (int, np.integer)) else int(s) for s in seeds]
        failed = np.zeros(self.n, dtype=bool)
        count = np.zeros(self.n, dtype=np.int64)
        step = -np.ones(self.n, dtype=np.int64)
        frontier = np.array(seeds, dtype=np.int64)
        failed[frontier] = True
        step[frontier] = 0
        rounds = 0
        while len(frontier):
            touched = []
            for f in frontier:
                nb = self.nbrs[f]
                count[nb] += 1
                touched.append(nb)
            cand = np.unique(np.concatenate(touched))
            cand = cand[~failed[cand]]
            new = cand[count[cand] >= need[cand]]
            if len(new) == 0:
                break
            rounds += 1
            failed[new] = True
            step[new] = rounds
            frontier = new
        return failed, step, rounds

    def size(self, seed, t, rule=">="):
        failed, _, rounds = self.run([seed], self.need(t, rule))
        return int(failed.sum()), rounds
