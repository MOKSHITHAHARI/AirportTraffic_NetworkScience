"""Automated tests of the simulator, driving the real Streamlit script with widget values (no browser needed).
Run with:  python3 app/test_simulator.py"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

SCRIPT = str(Path(__file__).resolve().parent / "simulator.py")


def metrics(at):
    out = {}
    for m in at.metric:
        v = m.value.replace("%", "")
        out[m.label] = int(v) if v.isdigit() else v
    return out


def fresh():
    at = AppTest.from_file(SCRIPT, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    return ok


def main():
    results = []
    at = fresh()
    m = metrics(at)
    results.append(check("default page loads (US, threshold model, ATL, 0.15): 541 airports failed", m["Airports failed"] == 541, str(m)))
    results.append(check("... in 13 rounds (the paper's 14 frames)", m["Rounds"] == 13))

    at.slider(key="threshold").set_value(0.20).run()
    m = metrics(at)
    results.append(check("threshold 0.20: 40 airports fail in 2 rounds (paper Fig. 14)", m["Airports failed"] == 40 and m["Rounds"] == 2, str(m)))

    at.radio(key="model").set_value("Load cascade (traffic re-routing and capacity)").run()
    at.selectbox(key="origin").set_value([o for o in at.selectbox(key="origin").options if o.startswith("ORD ")][0]).run()
    at.slider(key="alpha").set_value(0.10).run()
    m = metrics(at)
    results.append(check("load cascade, origin ORD, alpha 0.1: 48 failed (ORD + 47 overloaded), largest component 131", m["Airports failed"] == 48 and m["Largest connected component"] == 131, str(m)))
    results.append(check("... in 4 rounds", m["Rounds"] == 4))

    at.slider(key="alpha").set_value(0.5).run()
    m = metrics(at)
    results.append(check("load cascade with 50% spare capacity: only ORD fails", m["Airports failed"] == 1, str(m)))

    at.radio(key="model").set_value("Targeted attack (remove airports one by one)").run()
    at.selectbox(key="attack_kind").set_value("betweenness").run()
    at.slider(key="attack_k").set_value(5).run()
    m = metrics(at)
    results.append(check("targeted attack, betweenness, 5 airports: largest component 366 (paper Fig. 11)", m["Largest connected component"] == 366 and m["Airports failed"] == 5, str(m)))

    at.radio(key="protect_rule").set_value("top k by betweenness").run()
    at.slider(key="protect_k").set_value(5).run()
    m = metrics(at)
    results.append(check("protecting the top-5 betweenness airports changes the attack (LCC after 5 removals > 366)", m["Largest connected component"] > 366, str(m)))

    at2 = fresh()
    at2.selectbox(key="country").set_value("India").run()
    assert not at2.exception, at2.exception
    m = metrics(at2)
    results.append(check("India loads (71 airports) and the default failure runs", m["Airports failed"] >= 1, str(m)))
    results.append(check("India, threshold 0.15 from the biggest hub: complete collapse (critical threshold 0.33)", m["Airports failed"] == 71, str(m)))

    at3 = fresh()
    at3.radio(key="protect_rule").set_value("top k by degree").run()
    at3.slider(key="protect_k").set_value(10).run()
    m = metrics(at3)
    results.append(check("US, ATL protected (top 10 by degree), origin ATL: nothing fails", m["Airports failed"] == 0, str(m)))

    print(f"\n{sum(results)} of {len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
