"""Load-based cascading failures (Motter-Lai style) on the USFN.

Load of an airport = traffic routed through it, when every pair of airports exchanges traffic a_s * b_t along shortest
paths (split equally over equal-length paths). With a_s = b_t = 1 this is the number of shortest paths through the
airport, counting paths that start or end there. Capacity = (1 + alpha) x initial load. After a failure the traffic is
re-routed on the surviving network, loads are recomputed, and overloaded airports fail, until nothing new fails."""
import numpy as np
from numba import njit


@njit(cache=True)
def _loads(indptr, indices, active, a, b, endpoints):
    n = len(indptr) - 1
    load = np.zeros(n)
    dist = np.empty(n, dtype=np.int64)
    sigma = np.empty(n)
    delta = np.empty(n)
    order = np.empty(n, dtype=np.int64)
    for s in range(n):
        if not active[s]:
            continue
        for i in range(n):
            dist[i] = -1
            sigma[i] = 0.0
            delta[i] = 0.0
        dist[s] = 0
        sigma[s] = 1.0
        order[0] = s
        head = 0
        tail = 1
        while head < tail:
            v = order[head]
            head += 1
            for k in range(indptr[v], indptr[v + 1]):
                w = indices[k]
                if not active[w]:
                    continue
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    order[tail] = w
                    tail += 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
        reach_b = 0.0
        for idx in range(1, tail):
            reach_b += b[order[idx]]
        if endpoints:
            load[s] += a[s] * reach_b                               # s as a source
        for idx in range(tail - 1, 0, -1):
            w = order[idx]
            coeff = (b[w] + delta[w]) / sigma[w]
            for k in range(indptr[w], indptr[w + 1]):
                v = indices[k]
                if active[v] and dist[v] == dist[w] - 1:
                    delta[v] += sigma[v] * coeff
            if endpoints:
                load[w] += a[s] * b[w]                              # w as a target
            load[w] += a[s] * delta[w]                              # w as an intermediate node
    return load


class LoadCascade:
    def __init__(self, G, demand=None, endpoints=True):
        """demand: dict node -> weight (traffic generated and attracted by each airport); None = unit demand."""
        self.nodes = list(G.nodes)
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        self.n = len(self.nodes)
        ptr, ind = [0], []
        for n in self.nodes:
            ind.extend(sorted(self.idx[m] for m in G.neighbors(n)))
            ptr.append(len(ind))
        self.indptr = np.array(ptr, dtype=np.int64)
        self.indices = np.array(ind, dtype=np.int64)
        w = np.ones(self.n) if demand is None else np.array([float(demand[n]) for n in self.nodes])
        self.a = w
        self.b = w
        self.endpoints = bool(endpoints)
        self.deg = np.diff(self.indptr)
        self.L0 = self.loads(np.ones(self.n, dtype=np.bool_))

    def loads(self, active):
        """Ordered-pair loads on the subnetwork of active airports (inactive ones get 0)."""
        return _loads(self.indptr, self.indices, np.asarray(active, dtype=np.bool_), self.a, self.b, self.endpoints)

    def lcc_size(self, active):
        seen = ~np.asarray(active)
        best = 0
        for s in range(self.n):
            if seen[s]:
                continue
            stack, size = [s], 0
            seen[s] = True
            while stack:
                v = stack.pop()
                size += 1
                for k in range(self.indptr[v], self.indptr[v + 1]):
                    w = self.indices[k]
                    if not seen[w]:
                        seen[w] = True
                        stack.append(w)
            best = max(best, size)
        return best

    def capacity(self, alpha, mode="proportional"):
        """(1 + alpha) x initial load; airports that carry no load at all are given unlimited capacity."""
        cap = (1 + alpha) * self.L0
        cap[self.L0 <= 0] = np.inf
        return cap

    def run(self, seed, capacity, tol=1e-9, max_rounds=200):
        """Fail `seed`, then iterate re-routing and overload failures. Returns dict with the surviving mask,
        number of rounds and the damage split into direct (disconnection) and overload components."""
        seed = self.idx[seed] if not isinstance(seed, (int, np.integer)) else int(seed)
        active = np.ones(self.n, dtype=np.bool_)
        active[seed] = False
        rounds, history, fail_lists = 0, [], []
        while rounds < max_rounds:
            L = self.loads(active)
            over = active & (L > capacity * (1 + tol) + tol)
            if not over.any():
                break
            rounds += 1
            history.append(int(over.sum()))
            fail_lists.append(np.where(over)[0])
            active[over] = False
        return {"active": active, "rounds": rounds, "n_overload": int(self.n - 1 - active.sum()), "history": history, "failed_by_round": fail_lists, "lcc": self.lcc_size(active)}

    def capacity_het(self, alpha, u):
        """Heterogeneous tolerance: airport i gets capacity L0_i (1 + alpha * u_i) with u_i drawn once (mean 1)."""
        cap = self.L0 * (1 + alpha * u)
        cap[self.L0 <= 0] = np.inf
        return cap

    def capacity_additive(self, alpha):
        """Every airport gets the same absolute spare capacity: alpha x the mean initial load."""
        cap = self.L0 + alpha * self.L0[self.L0 > 0].mean()
        cap[self.L0 <= 0] = np.inf
        return cap

    def sweep(self, alphas, cap_fn=None, seeds=None):
        """For every origin and every alpha: size of the largest component after the cascade. Returns (matrix, lcc0)
        where lcc0 is the largest component right after the origin is removed (the direct, capacity-independent damage)."""
        cap_fn = cap_fn or (lambda a: self.capacity(a))
        seeds = list(range(self.n)) if seeds is None else list(seeds)
        out = np.zeros((len(seeds), len(alphas)), dtype=int)
        lcc0 = np.zeros(len(seeds), dtype=int)
        for r, s in enumerate(seeds):
            act = np.ones(self.n, dtype=np.bool_)
            act[s] = False
            L1 = self.loads(act)
            lcc0[r] = self.lcc_size(act)
            for gi, a in enumerate(alphas):
                cap = cap_fn(a)
                if not (act & (L1 > cap * (1 + 1e-9) + 1e-9)).any():
                    out[r, gi] = lcc0[r]
                else:
                    out[r, gi] = self.run(s, cap)["lcc"]
        return out, lcc0
