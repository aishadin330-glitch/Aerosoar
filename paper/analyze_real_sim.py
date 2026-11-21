"""Summary of the experiments re-run with the recorded sensor rates and noise (REAL=1 in gui/tests/paper_rev.mjs).
usage (project root): .venv/Scripts/python.exe paper/analyze_real_sim.py   -> paper/results/real_sim_summary.json"""
import json, numpy as np
from collections import defaultdict

R = "E:/AeroSoar/paper/results/"
L = lambda n: json.load(open(R + n))
rng = np.random.default_rng(1)
out = {}

# pipelines and detectors at nominal settings under the recorded sensors
st = defaultdict(list)
for r in L("rev_stress_real.json"):
    st[r["cond"]].append(r)
tab = {}
for c, v in st.items():
    d = {k: float(np.sqrt(np.mean([r["rmse"][k] ** 2 for r in v]))) for k in "ABC"}
    d["biasC"] = float(np.mean([r["bias"]["C"] for r in v]))
    T = sum(r["det"]["t"] for r in v) / 3600
    enc = sum(r["det"]["enc"] for r in v)
    for k in ("thr|1", "fix|1.5", "ada|1.5"):
        d[k] = dict(fa=sum(r["det"]["det"][k]["fa"] for r in v) / T, pd=sum(r["det"]["det"][k]["hit"] for r in v) / enc)
    tab[c] = d
out["stress"] = tab

# operating curves: choose on tuning flights (<=5 false alarms per hour), score on test flights
roc = L("rev_roc_real.json")
keys = list(roc[0]["det"].keys())
res = {}
for turb in (0.2, 0.5, 0.9):
    def rate(split, k):
        rs = [r for r in roc if r["split"] == split and r["turb"] == turb]
        enc = sum(r["enc"] for r in rs); T = sum(r["t"] for r in rs) / 3600
        return sum(r["det"][k]["hit"] for r in rs) / enc, sum(r["det"][k]["fa"] for r in rs) / T
    for fam in ("thr", "fix", "ada"):
        best = None
        for k in [k for k in keys if k.startswith(fam)]:
            pd, fa = rate("tune", k)
            if fa <= 5 and (best is None or pd > rate("tune", best)[0]):
                best = k
        if best is None:
            res[f"{turb}|{fam}"] = None
            continue
        rs = [r for r in roc if r["split"] == "test" and r["turb"] == turb]
        enc = np.array([r["enc"] for r in rs]); hit = np.array([r["det"][best]["hit"] for r in rs]); fa = np.array([r["det"][best]["fa"] for r in rs]); T = np.array([r["t"] for r in rs]) / 3600
        P = [],
        P, F = [], []
        for _ in range(2000):
            i = rng.integers(0, len(rs), len(rs))
            P.append(hit[i].sum() / enc[i].sum()); F.append(fa[i].sum() / T[i].sum())
        res[f"{turb}|{fam}"] = dict(k=best, pd=[hit.sum() / enc.sum(), *np.percentile(P, [2.5, 97.5])], fa=[fa.sum() / T.sum(), *np.percentile(F, [2.5, 97.5])])
out["roc"] = res

# closed loop cutoff sweep
g = defaultdict(list)
for r in L("rev_latency_fc_real.json"):
    g[r["name"]].append(r)
out["fc"] = {k: dict(speed=float(np.mean([r["speed"] for r in v])), rej=float(np.mean([r["rejected"] for r in v])),
                     ctr=float(np.nanmean([r["ctr"] for r in v if r["ctr"] is not None])), arr=sum(r["arrived"] for r in v)) for k, v in g.items()}

# Kalman
kal = defaultdict(list)
for r in L("rev_kalman_on_real.json"):
    kal[(r["baro"], r["qb"], r["qj"])].append(r)
pool = lambda v, a, n: float(np.sqrt(sum((r[a] or 0) ** 2 * r[n] for r in v) / max(1, sum(r[n] for r in v))))
out["kalman"] = {str(k): dict(steady=pool(v, "steady", "ns"), manoeuvre=pool(v, "manoeuvre", "nm")) for k, v in kal.items()}
json.dump(out, open(R + "real_sim_summary.json", "w"))

print("STRESS (recorded sensors)")
for c, d in tab.items():
    print(" ", c, {k: (round(x, 3) if isinstance(x, float) else {a: round(b, 3) for a, b in x.items()}) for k, x in d.items()})
print("ROC")
for k, v in res.items():
    print(" ", k, None if v is None else (v["k"], [round(x, 3) for x in v["pd"]], [round(x, 2) for x in v["fa"]]))
print("FC")
for k, v in out["fc"].items():
    print(" ", k, {a: round(b, 2) for a, b in v.items()})
print("KALMAN (no baro drift)")
for k, v in out["kalman"].items():
    if k.startswith("(False"):
        print(" ", k, {a: round(b, 3) for a, b in v.items()})
