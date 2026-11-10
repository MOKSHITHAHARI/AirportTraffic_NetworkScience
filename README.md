# Airport Traffic Network Science

A reproduction, critique and extension of **"Analysis and Resilience of the U.S. Flight Network"** (Pandey & Kafle, [arXiv:2505.11559](https://arxiv.org/abs/2505.11559)), built on the open [OpenFlights](https://openflights.org/data.html) route data.

The project does two things:

1. **Part A: Reproduce the paper.** Rebuild the U.S. Flight Network (541 airports, 2,780 weighted edges) and match the paper's reported results: structure, power-law fit, communities, null models, targeted attacks and threshold cascades.
2. **Part B: Go beyond it.** Test the paper's assumptions and conclusions with weighted analysis, real traffic data, a load-based cascade model, 24 countries, protection strategies and an airline-level multilayer view. An interactive simulator lets you explore the failure models yourself.

---

## Key findings

**Reproduced exactly**
- Network size and hub: 541 airports, 2,780 edges, ATL has the highest degree (153).
- L = 3.2, diameter 7, average degree 10, ANC has the highest betweenness.
- The power-law exponent γ = 1.89, the Louvain community tables, the ER/BA comparison (Table 3), the attack results (Table 4, Fig. 11) and the cascade figures (Figs. 12 to 14).

**Where the analysis goes further**

| Topic | Finding |
|---|---|
| The 541 nodes | They are the largest connected component of a 549-airport domestic graph. The paper does not say this. |
| Edge weights | The paper builds weights but never uses them. 44% of the route rows are codeshare listings. |
| Power law | The paper's recipe (continuous MLE) overestimates γ on integer data. The discrete estimate is about 1.73. A power law with a cutoff fits better than a pure power law. |
| Communities | One Louvain community is exactly Alaska. A single run is one draw: Louvain gives 5 to 13 communities depending on the seed. |
| Null models | A degree-preserving null reproduces about 85% of the clustering excess, so much of the "community structure" evidence is the hub-and-spoke degree sequence itself. |
| Attacks | "Betweenness is most destructive" holds only in the paper's 5-removal window. Adaptive betweenness attacks are the most damaging overall. |
| Threshold cascade | The collapse threshold is a knife-edge near 1/6. ATL is **not** the worst origin; LAS is. |
| Load cascade | Dangerous origins are high-load hubs (ORD, DEN, LAS, MSP, LAX). The two cascade models agree only partly (Spearman 0.38), so the paper's cascade conclusion is model-dependent. |
| Other countries | The US result is not a general law: it has the lowest γ among 24 countries. |
| Protection | Protecting hubs works, and betweenness, degree and PageRank are the best rules. Adding routes is the weaker lever. |
| Airlines | 68% of routes are served by a single airline. Losing one small airline (Allegiant) can disconnect 32 airports. |

---

## Repository structure

```
.
├── data/raw/            OpenFlights files: airports, routes, airlines, planes (.dat)
├── src/                 Reusable analysis modules (used by notebooks and the app)
│   ├── usfn.py            Load data, build the weighted network, basic stats
│   ├── powerlaw_tools.py  Discrete MLE, bootstrap, goodness of fit, model comparison
│   ├── communities.py     Louvain, label propagation and partition comparison
│   ├── nullmodels.py      ER, BA and degree-preserving rewiring
│   ├── attacks.py         Static and adaptive targeted attacks, percolation measures
│   ├── cascade.py         Threshold cascade model (the paper's)
│   ├── load_cascade.py    Motter-Lai load-capacity cascade
│   ├── weighted.py        Weighted metrics (strength, disparity, weighted paths)
│   ├── regions.py         Build networks for any country or region
│   └── resilience.py      Protection, rewiring and recovery strategies
├── notebooks/           One notebook per milestone (see below)
├── app/
│   ├── simulator.py       Streamlit failure simulator
│   └── test_simulator.py  Tests for the simulator engine
└── requirements.txt
```

## Notebooks

Each notebook states its goal, what the paper claims, the result and a short summary. Part A notebooks end with assertions that check the numbers against the paper.

| Notebook | Topic | Paper section |
|---|---|---|
| `M00_data_and_network` | Data card, filtering funnel, building the network | 2 |
| `M01_basic_structure` | Path lengths, degree distribution, betweenness | 3 |
| `M02_power_law` | Reproducing γ = 1.89 and testing the recipe | 4.1 |
| `M03_communities` | Louvain vs label propagation, validation | 4.2 |
| `M04_null_models` | ER, BA and degree-preserving nulls | 4.3 |
| `M05_attacks` | Random failure vs targeted attack, adaptive attacks | 5.1 |
| `M06_cascade` | Threshold cascade, critical threshold, worst-case origins | 5.2 |
| `M07_weighted` | Weighted analysis and the codeshare caveat | Part B |
| `M08_traffic_delays` | Real BTS flight traffic and delays | Part B |
| `M09_load_cascade` | Load-capacity cascade | Part B |
| `M10_countries` | 24-country comparison, India vs US | Part B |
| `M11_protection` | Protection, rewiring and recovery | Part B |
| `M12_multilayer` | Multilayer network by airline | Part B |

---

## Getting started

### Install

```bash
git clone <your-repo-url>
cd AirportTraffic_NetworkScience
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`leidenalg` and `igraph` ship compiled wheels for common platforms. If the install fails on your system, only the Leiden comparison in M03 is affected.

### Run the notebooks

```bash
jupyter lab notebooks/
```

Run them from the repository root or the `notebooks/` folder. The Part A notebooks (M00 to M06) are the best starting point and need only the bundled data. Some cells run bootstrap and simulation loops, so a few notebooks take several minutes.

> **Note on M08:** it uses 2.7 million BTS on-time flight records from `data/kaggle/`, which is **not included** in this repository because of its size. Download the dataset separately and place it in `data/kaggle/` to run that notebook. The other notebooks do not need it.

### Run the simulator

```bash
streamlit run app/simulator.py
```

In the app you can:
- choose one of 24 countries,
- pick a failure model: the paper's **threshold cascade**, the **load cascade**, or a **targeted attack**,
- set the origin airport or click one on the map,
- protect chosen airports,
- step through the failure round by round and watch the largest connected component shrink.

### Run the tests

```bash
python app/test_simulator.py
```

The script drives the real Streamlit app with Streamlit's `AppTest` harness, so no browser is needed.

---

## Data

| File | Source | Use |
|---|---|---|
| `airports.dat`, `routes.dat`, `airlines.dat`, `planes.dat` | [OpenFlights](https://openflights.org/data.html) | Network construction |
| BTS on-time performance records (not included) | U.S. Bureau of Transportation Statistics | Real traffic and delays (M08) |

Caveats worth knowing:
- OpenFlights routes are a **2014 snapshot**; the source stopped updating in June 2014.
- One route row is one airline serving one directed airport pair. It is not a flight count or a passenger count. About 44% of rows are codeshare listings.
- The paper used the Kaggle *Global Air Transportation Network* dataset, a repackaging of OpenFlights. The U.S. network built here is identical to the paper's.

## Method notes

- **Edge weight:** number of route rows between two airports, as in the paper. M07 also uses an operated-only version to bound the codeshare effect.
- **Largest connected component:** all metrics use the 541-airport LCC so that path lengths and diameter are defined.
- **Threshold cascade:** a node fails when the failed fraction of its neighbours is at least the threshold (the `>=` rule is the one that matches the paper).
- **Load cascade:** Motter-Lai model. A node carries all shortest-path traffic through it and fails above (1 + α) times its original load.

## Reference

Pandey, B. & Kafle, S. *Analysis and Resilience of the U.S. Flight Network.* arXiv:2505.11559.

Other methods used: Clauset, Shalizi & Newman (2009) for power-law testing; Motter & Lai (2002) for the load cascade.

## Authors

Add your names, course and institution here.

## License

Add a license. The OpenFlights data has its own terms (Open Database License); check them before redistributing.
