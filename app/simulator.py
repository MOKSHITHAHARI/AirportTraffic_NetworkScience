"""Airport network failure simulator (M13).

Run with:  streamlit run app/simulator.py

Pick a country, a failure model (threshold cascade of the paper, load cascade, or targeted attack), an origin airport
(or click one on the map), optionally protect some airports, and step through the failure round by round.
Uses the same code (src/) as the milestone notebooks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import cascade as cs  # noqa: E402
import load_cascade as lc  # noqa: E402
import regions as rg  # noqa: E402
import resilience as rs  # noqa: E402
import usfn  # noqa: E402

COUNTRIES = ["United States", "Canada", "China", "Brazil", "Australia", "Russia", "India", "Indonesia", "Japan", "Mexico", "United Kingdom", "Norway",
             "Colombia", "France", "Turkey", "Iran", "Spain", "Greece", "Philippines", "Argentina", "Sweden", "Italy", "Thailand", "Malaysia"]
MODELS = ["Threshold cascade (the paper's model)", "Load cascade (traffic re-routing and capacity)", "Targeted attack (remove airports one by one)"]
BLUE, RED, DARK, ORANGE, GREY, GREEN = "#3b6ea5", "#c05746", "#4a0d0d", "#f2a900", "#b8b8b8", "#2e8b57"

st.set_page_config(page_title="Airport network failure simulator", layout="wide")


# ------------------------------------------------------------------ data and engines (cached)
@st.cache_resource(show_spinner="Loading OpenFlights data...")
def load_data():
    return usfn.load_openflights()


@st.cache_resource(show_spinner="Building the network...")
def get_network(country):
    airports, routes = load_data()
    g = rg.build_region_network(airports, routes, [country])
    nodes = list(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows = [idx[u] for u, v in g.edges] + [idx[v] for u, v in g.edges]
    cols = [idx[v] for u, v in g.edges] + [idx[u] for u, v in g.edges]
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return {"g": g, "nodes": nodes, "idx": idx, "n": n, "adj": adj, "deg": np.array([g.degree(v) for v in nodes]),
            "label": [f"{v} - {g.nodes[v]['city']}" for v in nodes], "xy": np.array([[g.nodes[v]["lon"], g.nodes[v]["lat"]] for v in nodes]),
            "edges": np.array([[idx[u], idx[v]] for u, v in g.edges]), "threshold": cs.Cascade(g)}


@st.cache_resource(show_spinner="Preparing the load model (one-off)...")
def get_load_engine(country):
    return lc.LoadCascade(get_network(country)["g"])


def components(net, active):
    """Component label of every active airport (-1 for failed) and the size of the largest component."""
    if not active.any():
        return -np.ones(net["n"], dtype=int), -1, 0
    sub = net["adj"][active][:, active]
    _, lab = connected_components(sub, directed=False)
    full = -np.ones(net["n"], dtype=int)
    full[np.where(active)[0]] = lab
    sizes = np.bincount(lab) if len(lab) else np.array([0])
    main = int(np.argmax(sizes)) if len(lab) else -1
    return full, main, int(sizes.max()) if len(lab) else 0


@st.cache_data(show_spinner="Ranking airports for the attacker...")
def cached_attack_order(country, kind, protected):
    net = get_network(country)
    return rs.attack_order(net["g"], [net["nodes"][i] for i in protected], kind, net["nodes"])


# ------------------------------------------------------------------ simulation: a list of "rounds", each an array of airport indices
def simulate(net, country, model, origin, protected, threshold, alpha, attack_kind, attack_k):
    n = net["n"]
    prot = np.zeros(n, dtype=bool)
    prot[list(protected)] = True
    if model.startswith("Threshold"):
        C = net["threshold"]
        need = C.need(threshold).copy()
        need[prot] = n + 1                                           # protected airports can never fail
        if prot[origin]:
            return [np.array([], dtype=int)]
        failed, step, rounds = C.run([origin], need)
        return [np.where(step == r)[0] for r in range(rounds + 1)]
    if model.startswith("Load"):
        LC = get_load_engine(country)
        cap = LC.capacity(alpha).copy()
        cap[prot] = np.inf                                           # protected airports have unlimited capacity
        if prot[origin]:
            return [np.array([], dtype=int)]
        r = LC.run(origin, cap)
        return [np.array([origin])] + [np.asarray(f) for f in r["failed_by_round"]]
    order = cached_attack_order(country, attack_kind, tuple(sorted(protected)))
    return [np.array([i]) for i in order[:attack_k]]


# ------------------------------------------------------------------ page
st.title("Airport network failure simulator")
st.caption("Companion to the project that reproduces and extends 'Analysis and Resilience of the U.S. Flight Network' (Pandey and Kafle, 2025). "
           "OpenFlights data (2014 snapshot); idealised models, not a forecast.")

with st.sidebar:
    st.header("Network")
    country = st.selectbox("Country", COUNTRIES, index=0, key="country")
    net = get_network(country)
    st.caption(f"{net['n']} airports, {net['g'].number_of_edges()} routes (domestic, largest connected component)")

    st.header("Failure model")
    model = st.radio("Model", MODELS, index=0, key="model")
    threshold = alpha = None
    attack_kind, attack_k = "betweenness", 10
    if model.startswith("Threshold"):
        threshold = st.slider("Breakdown threshold", 0.05, 0.50, 0.15, 0.005, key="threshold",
                              help="An airport fails when this fraction of its neighbours has failed (the paper's rule: failed/degree >= threshold).")
    elif model.startswith("Load"):
        alpha = st.slider("Spare capacity alpha", 0.0, 1.0, 0.10, 0.01, key="alpha",
                          help="Each airport can carry (1 + alpha) times its normal traffic; failed airports' traffic is re-routed.")
        if net["n"] > 600:
            st.warning("The load model is only offered for networks up to 600 airports.")
    else:
        attack_kind = st.selectbox("Attacker", ["degree", "betweenness", "betweenness_adaptive"], index=1, key="attack_kind",
                                   format_func=lambda s: {"degree": "by degree (ranked once)", "betweenness": "by betweenness (ranked once)", "betweenness_adaptive": "by betweenness, re-ranked after each removal"}[s])
        attack_k = st.slider("Airports removed", 1, min(60, net["n"] - 1), 10, key="attack_k")

    order_deg = list(np.argsort(-net["deg"], kind="stable"))
    labels_sorted = [net["label"][i] for i in order_deg]
    if not model.startswith("Targeted"):
        st.header("Origin of the failure")
        if "origin" not in st.session_state or st.session_state.get("origin_country") != country:
            st.session_state["origin"] = labels_sorted[0]
            st.session_state["origin_country"] = country
        sel = st.session_state.get("map")
        if sel and sel.get("selection", {}).get("points"):
            cd = sel["selection"]["points"][0].get("customdata")
            cd = cd[0] if isinstance(cd, (list, tuple)) else cd
            if cd in net["idx"] and net["label"][net["idx"][cd]] != st.session_state["origin"]:
                st.session_state["origin"] = net["label"][net["idx"][cd]]
        origin_label = st.selectbox("Airport that fails first (or click it on the map)", labels_sorted, key="origin")
        origin = net["label"].index(origin_label)
    else:
        origin = 0

    st.header("Protection")
    rule = st.radio("Harden airports (they cannot be attacked or fail)", ["none", "top k by degree", "top k by betweenness"], index=0, key="protect_rule")
    protected = []
    if rule != "none":
        k_p = st.slider("Number of protected airports", 1, 40, 10, key="protect_k")
        if rule.endswith("degree"):
            protected = [int(i) for i in order_deg[:k_p]]
        else:
            protected = [int(i) for i in cached_attack_order(country, "betweenness", ())[:k_p]]
    extra = st.multiselect("Also protect", labels_sorted, key="protect_extra")
    protected = sorted(set(protected) | {net["label"].index(x) for x in extra})
    if protected:
        st.caption("Protected: " + ", ".join(net["nodes"][i] for i in protected[:12]) + (" ..." if len(protected) > 12 else ""))

rounds = simulate(net, country, model, origin, protected, threshold, alpha, attack_kind, attack_k) if not (model.startswith("Load") and net["n"] > 600) else [np.array([], dtype=int)]
n_rounds = len(rounds) - 1
if n_rounds > 0:
    step_round = st.slider("Step through the failure (round)", 0, n_rounds, n_rounds)
else:
    step_round = 0
    st.info("Nothing fails: the failure does not spread (or the origin is protected).")

failed = np.zeros(net["n"], dtype=bool)
for r in range(step_round + 1):
    failed[rounds[r]] = True
new_now = np.zeros(net["n"], dtype=bool)
new_now[rounds[step_round]] = True
labels_c, main, lcc = components(net, ~failed)

cols = st.columns(4)
cols[0].metric("Airports failed", int(failed.sum()), help="Includes the origin (or the removed airports).")
cols[1].metric("Largest connected component", lcc, help="Size of the biggest group of airports that can still reach each other.")
cols[2].metric("Rounds" if not model.startswith("Targeted") else "Removal steps", n_rounds if not model.startswith("Targeted") else len(rounds), help="Number of rounds the failure took to stop (or airports removed in the attack).")
cols[3].metric("Share of the network still connected", f"{lcc / net['n']:.0%}")

# ---- map
xy, edges = net["xy"], net["edges"]
alive_e = edges[(~failed[edges[:, 0]]) & (~failed[edges[:, 1]])]
ex, ey = [], []
for a, b in alive_e:
    ex += [xy[a, 0], xy[b, 0], None]; ey += [xy[a, 1], xy[b, 1], None]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(110,110,110,0.25)", width=0.6), hoverinfo="skip", name="routes"))
state = np.array(["connected"] * net["n"], dtype=object)
state[(~failed) & (labels_c != main)] = "cut off from the main network"
state[failed] = "failed earlier"; state[new_now & failed] = "failed this round"
if not model.startswith("Targeted") and failed[origin]:
    state[origin] = "origin"
colors = {"connected": BLUE, "cut off from the main network": GREY, "failed earlier": RED, "failed this round": DARK, "origin": ORANGE}
for s_, c_ in colors.items():
    ids = np.where(state == s_)[0]
    if len(ids) == 0:
        continue
    fig.add_trace(go.Scatter(x=xy[ids, 0], y=xy[ids, 1], mode="markers", name=f"{s_} ({len(ids)})", customdata=[net["nodes"][i] for i in ids],
                             marker=dict(size=5 + 2.2 * np.sqrt(net["deg"][ids]), color=c_, line=dict(width=[3 if i in set(protected) else 0.6 for i in ids], color=[GREEN if i in set(protected) else "white" for i in ids])),
                             text=[f"{net['label'][i]}<br>{net['g'].nodes[net['nodes'][i]]['name']}<br>connections: {net['deg'][i]}" for i in ids], hoverinfo="text"))
fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.05), xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1.35),
                  plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="map")
if protected:
    st.caption("Green rings mark protected airports.")

# ---- per-round table and chart
rows = []
cum = np.zeros(net["n"], dtype=bool)
lcc_by_round = []
for r, f in enumerate(rounds):
    cum[f] = True
    _, _, l = components(net, ~cum)
    lcc_by_round.append(l)
    rows.append({"round": r, "airports failing": len(f), "total failed": int(cum.sum()), "largest component": l,
                 "examples": ", ".join(net["nodes"][i] for i in f[:8]) + (" ..." if len(f) > 8 else "")})
left, right = st.columns([3, 2])
with left:
    st.subheader("Round by round")
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
with right:
    st.subheader("Largest component by round")
    st.line_chart(pd.DataFrame({"largest component": lcc_by_round}, index=range(len(lcc_by_round))))

round_of = {net["nodes"][i]: r for r, f in enumerate(rounds) for i in f}
csv = pd.DataFrame({"airport": list(round_of), "failed in round": list(round_of.values())}).to_csv(index=False)
st.download_button("Download the failure sequence (CSV)", csv, file_name="failure_sequence.csv", mime="text/csv")

with st.expander("About the models"):
    st.markdown("""
**Threshold cascade (the paper's model, milestone M6).** An airport fails when the fraction of its neighbours that have failed is at least the breakdown threshold. For the US, failing ATL gives a complete collapse at threshold 0.15 and a contained failure (40 airports) at 0.20.

**Load cascade (M9).** Every pair of airports exchanges traffic along shortest paths; an airport can carry (1 + alpha) times its initial load. When an airport fails, traffic is re-routed, and overloaded airports fail in turn.

**Targeted attack (M5).** Airports are removed one by one, ranked by degree or betweenness (once, or re-ranked after each removal).

**Protection (M11).** A protected airport cannot be removed, cannot fail, and has unlimited capacity. Hardening the top airports by betweenness is far more effective than choosing them by degree or at random.

Data: OpenFlights routes (2014 snapshot, crowd-sourced); domestic routes only; no traffic information. See the project documentation for limits.
""")
