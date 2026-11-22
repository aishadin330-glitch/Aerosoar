"""Compare the pipelines replayed on recorded ALFA flights (gui/tests/real_replay.mjs) with an independent GNSS-based reference.
usage (project root): .venv/Scripts/python.exe paper/analyze_real_replay.py  -> paper/results/real_replay_summary.json

There is no true vertical air velocity in recorded data, and the aircraft is powered, so the quantity compared is the energy rate dE/dt = dh/dt + V dV/dt / g
(the part of w that does not need the drag polar or the thrust). Each pipeline's estimate is its output minus its sink term.
Reference: GNSS Doppler vertical velocity plus (V/g) times the speed rate of the pitot signal, both smoothed with a zero-phase 0.5 Hz Butterworth filter,
so the reference has no delay. Its own error (GNSS noise) is common to all pipelines. Strata: speed change (|dV/dt| >= 0.15 m/s^2 in the reference) and steady."""
import json, pathlib
import numpy as np
from scipy import signal, stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = 9.80665
FS = 10.0
segs = json.load(open(ROOT / "data/alfa/streams/segments.json"))
rep = json.load(open(ROOT / "data/alfa/streams/replay_out.json"))
b_lp, a_lp = signal.butter(2, 0.5 / (FS / 2))
zp = lambda x: signal.filtfilt(b_lp, a_lp, x)
tq = lambda n: stats.t.ppf(0.975, n - 1)

# ---- reference series per segment
ref = []
lags, signs = [], []
for s, r in zip(segs, rep):
    t = np.array(r["t"])
    Vp = np.interp(t, s["arsp"]["t"], s["arsp"]["v"])
    hb = np.interp(t, s["baro"]["t"], s["baro"]["alt"])
    vz_raw = np.interp(t, s["gps"]["t"], s["gps"]["vz"])
    clb = zp(signal.savgol_filter(hb, 21, 2, deriv=1, delta=1 / FS))
    # GNSS velocity: find the sign and the lag that make it agree with the barometric climb rate
    best = None
    for lag in range(-10, 11):
        v = np.roll(vz_raw, -lag)
        c = np.corrcoef(zp(v)[30:-30], clb[30:-30])[0, 1]
        if best is None or abs(c) > abs(best[0]):
            best = (c, lag)
    signs.append(np.sign(best[0])); lags.append(best[1])
    ref.append(dict(t=t, Vp=Vp, vz=vz_raw, clb=clb))
sign = float(np.median(signs)); lag = int(np.median(lags))
print("GNSS vz sign", sign, "lag samples", lag, "(positive = GNSS later)")

pipes = {"B": ("B", "sB"), "C": ("C", "sC"), "CL": ("CL", "sC"), "CS": ("CS", "sC"), "D": ("D", "sB"), "E": ("E", "sB")}
res = {p: {"steady": [], "man": []} for p in pipes}
diff = {"C-CL": {"steady": [], "man": []}, "C-CS": {"steady": [], "man": []}, "B-CL": {"steady": [], "man": []}}
pool = {p: {"steady": [0, 0], "man": [0, 0]} for p in pipes}
poold = {k: {"steady": [0, 0], "man": [0, 0]} for k in diff}
nstr = {"steady": 0, "man": 0}
refq = []
for rf, r in zip(ref, rep):
    vz = np.roll(rf["vz"], -lag) * sign
    dV = zp(signal.savgol_filter(rf["Vp"], 21, 2, deriv=1, delta=1 / FS))
    E = zp(vz) + zp(rf["Vp"]) * dV / G
    man = np.abs(dV) >= 0.15
    ok = np.ones(len(E), bool); ok[:30] = False; ok[-30:] = False
    est = {p: np.array(r[k]) - np.array(r[sk]) for p, (k, sk) in pipes.items()}
    for p in pipes:
        for st, m in (("steady", ~man & ok), ("man", man & ok)):
            if m.sum() >= 20:
                e = est[p][m] - E[m]
                res[p][st].append(float(np.sqrt(np.mean(e ** 2))))
                pool[p][st][0] += float(np.sum(e ** 2)); pool[p][st][1] += int(m.sum())
    for k, (x, y) in {"C-CL": ("C", "CL"), "C-CS": ("C", "CS"), "B-CL": ("B", "CL")}.items():
        for st, m in (("steady", ~man & ok), ("man", man & ok)):
            if m.sum() >= 20:
                d = est[x][m] - est[y][m]
                diff[k][st].append(float(np.sqrt(np.mean(d ** 2))))
                poold[k][st][0] += float(np.sum(d ** 2)); poold[k][st][1] += int(m.sum())
    nstr["steady"] += int((~man & ok).sum()); nstr["man"] += int((man & ok).sum())
    refq.append(float(np.sqrt(np.mean((zp(vz)[ok] - rf["clb"][ok]) ** 2))))

mci = lambda x: (float(np.mean(x)), float(tq(len(x)) * np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else (float(np.mean(x)), float("nan"))
out = {"n_segments": len(segs), "samples": nstr, "minutes": round((nstr["steady"] + nstr["man"]) / FS / 60, 1), "sign": sign, "lag": lag,
       "pipes": {p: {st: dict(per_segment=mci(v[st]), pooled=float(np.sqrt(pool[p][st][0] / pool[p][st][1])), n_seg=len(v[st])) for st in ("steady", "man")} for p, v in res.items()},
       "diff": {k: {st: dict(per_segment=mci(v[st]), pooled=float(np.sqrt(poold[k][st][0] / poold[k][st][1]))) for st in ("steady", "man")} for k, v in diff.items()},
       "ref_vs_baro_rms": float(np.mean(refq))}
(ROOT / "paper" / "results" / "real_replay_summary.json").write_text(json.dumps(out))
print("segments", out["n_segments"], "minutes scored", out["minutes"], "samples", nstr, "share speed-change %.1f%%" % (100 * nstr["man"] / (nstr["man"] + nstr["steady"])))
print("reference vs barometric climb rate RMS (m/s): %.3f" % out["ref_vs_baro_rms"])
for p, v in out["pipes"].items():
    print("  %-3s steady %.3f+-%.3f (pooled %.3f)   speed change %.3f+-%.3f (pooled %.3f)   n_seg %d/%d" % (p, *v["steady"]["per_segment"], v["steady"]["pooled"], *v["man"]["per_segment"], v["man"]["pooled"], v["steady"]["n_seg"], v["man"]["n_seg"]))
for k, v in out["diff"].items():
    print("  diff %-5s steady %.3f+-%.3f   speed change %.3f+-%.3f" % (k, *v["steady"]["per_segment"], *v["man"]["per_segment"]))

# ---- error by speed-rate activity bin (pooled over segments)
edges = [0, 0.05, 0.1, 0.15, 0.3, 10]
acc = {p: [[0.0, 0] for _ in range(len(edges) - 1)] for p in pipes}
for rf, r in zip(ref, rep):
    vz = np.roll(rf["vz"], -lag) * sign
    dV = zp(signal.savgol_filter(rf["Vp"], 21, 2, deriv=1, delta=1 / FS))
    E = zp(vz) + zp(rf["Vp"]) * dV / G
    ok = np.ones(len(E), bool); ok[:30] = False; ok[-30:] = False
    for p, (k, sk) in pipes.items():
        e = np.array(r[k]) - np.array(r[sk]) - E
        for i in range(len(edges) - 1):
            m = ok & (np.abs(dV) >= edges[i]) & (np.abs(dV) < edges[i + 1])
            acc[p][i][0] += float(np.sum(e[m] ** 2)); acc[p][i][1] += int(m.sum())
print("bins of |dV/dt| (m/s^2):", edges)
print("   samples:", [acc["B"][i][1] for i in range(len(edges) - 1)])
for p in pipes:
    print("  %-3s" % p, [round(float(np.sqrt(a / n)), 3) if n else None for a, n in acc[p]])
out["bins"] = {"edges": edges, "n": [acc["B"][i][1] for i in range(len(edges) - 1)], "rmse": {p: [float(np.sqrt(a / n)) if n else None for a, n in acc[p]] for p in pipes}}
(ROOT / "paper" / "results" / "real_replay_summary.json").write_text(json.dumps(out))
