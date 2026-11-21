"""DSP figures and tables for the paper.   usage (project root): .venv/Scripts/python.exe paper/make_dsp_results.py"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parent.parent
R, F = ROOT / "paper" / "results", ROOT / "paper" / "figures"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
D = json.load(open(R / "dsp.json"))
FS = 20.0


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


NAMES = {"butter": "Butterworth (2nd order)", "movavg": "Moving average", "ema": "One-pole"}
summary = {}

# ---------------------------------------------------------------- E1 filters: error against delay
print("FILTER SWEEP (30 flights each): RMSE m/s, measured delay s, theoretical delay s")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
rows = []
for t in NAMES:
    xs, ys, es = [], [], []
    for fc in [0.1, 0.2, 0.4, 0.8, 1.5]:
        r = [x for x in D["filters"] if x["type"] == t and x["fc"] == fc]
        rm = np.mean([x["rmse"] for x in r]); dl = np.mean([x["delay"] for x in r]); th = np.mean([x["delayTheory"] for x in r])
        xs.append(dl); ys.append(rm); es.append(ci([x["rmse"] for x in r]))
        rows.append((t, fc, rm, ci([x["rmse"] for x in r]), dl, th))
        print(f"{NAMES[t]:<26} fc={fc:<4} rmse {rm:.3f} +-{ci([x['rmse'] for x in r]):.3f}  delay {dl:.2f}  theory {th:.2f}")
    ax[0].errorbar(xs, ys, yerr=es, marker="o", ms=3, capsize=2, label=NAMES[t])
    for fc, x, y in zip([0.1, 0.2, 0.4, 0.8, 1.5], xs, ys):
        if t == "butter":
            ax[0].annotate(f"{fc}", (x, y), textcoords="offset points", xytext=(3, 4), fontsize=6)
ax[0].set(xlabel="Measured delay, s", ylabel="RMSE of vertical air velocity, m/s")
ax[0].legend(fontsize=6)
summary["filters"] = rows

# frequency response and group delay (analytic, same coefficients as the implementation)
fc = 0.4
K = np.tan(np.pi * fc / FS)
n = 1 / (1 + np.sqrt(2) * K + K * K)
b = np.array([K * K * n, 2 * K * K * n, K * K * n])
a = np.array([1, 2 * (K * K - 1) * n, (1 - np.sqrt(2) * K + K * K) * n])
N = round(0.443 * FS / fc)
al = 1 - np.exp(-2 * np.pi * fc / FS)
sysd = {"Butterworth": (b, a), "Moving average": (np.ones(N) / N, [1.0]), "One-pole": ([al], [1, -(1 - al)])}
for k, (bb, aa) in sysd.items():
    f, H = signal.freqz(bb, aa, worN=2048, fs=FS)
    ax[1].plot(f, 20 * np.log10(np.maximum(np.abs(H), 1e-4)), label=k)
ax[1].set(xlabel="Frequency, Hz", ylabel="Gain, dB", xlim=(0, 3), ylim=(-50, 3))
ax[1].axvline(fc, color="k", ls="--", lw=0.7)
ax[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(F / "filters.pdf")
fig.savefig(F / "filters.png")

# ---------------------------------------------------------------- ablation: total energy and Kalman fusion (full missions)
print("\nABLATION of the signal chain: RMSE m/s (30 missions, mean +- 95% CI); steady = |dV/dt| < 0.15 m/s^2")
ab = {}
for cond in ["nominal", "baro drift"]:
    for part in ["steady", "manoeuvre"]:
        r = [x[part] for x in D["ablation"] if x["cond"] == cond]
        ab[f"{cond}|{part}"] = {k: (float(np.mean([x[k] for x in r])), ci([x[k] for x in r])) for k in ["raw", "noTE", "te", "kf"]}
        ab[f"{cond}|{part}"]["n"] = float(np.mean([x["n"] for x in r]))
        print(cond, part, {k: (f"{v[0]:.3f}+-{v[1]:.3f}" if k != "n" else int(v)) for k, v in ab[f"{cond}|{part}"].items()})
summary["ablation"] = ab

# ---------------------------------------------------------------- E3 detectors
print("\nDETECTORS (30 flights x 300 s per turbulence level)")
dt = {}
for turb in [0.2, 0.5, 0.9]:
    for det in ["Threshold, 1 m/s for 2 s", "CUSUM, fixed", "CUSUM, gust-adaptive", "CUSUM on unfiltered w"]:
        r = [x for x in D["detectors"] if x["det"] == det and x["turb"] == turb]
        enc = sum(x["enc"] for x in r); hit = sum(x["hit"] for x in r); fa = sum(x["fa"] for x in r); hrs = sum(x["t"] for x in r) / 3600
        lat = [l for x in r for l in x["lat"]]
        dt[(turb, det)] = dict(enc=enc, hit=hit, rate=hit / enc, fa_per_h=fa / hrs, lat=float(np.median(lat)) if lat else float("nan"))
        print(f"turb {turb} {det:<28} detection {hit}/{enc} = {hit / enc:.2f}  false alarms/h {fa / hrs:6.1f}  median latency {dt[(turb, det)]['lat']:.1f} s")
summary["detectors"] = {f"{k[0]}|{k[1]}": v for k, v in dt.items()}

# ---------------------------------------------------------------- spectra
s = {k: np.array(v) for k, v in D["series"].items()}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
band = {}
for k, lab, c in [("w", "Truth", "k"), ("A", "A: unfiltered", "0.55"), ("B", "B: Butterworth", "tab:green"), ("C", "C: Kalman", "tab:blue")]:
    f, P = signal.welch(s[k][200:], fs=FS, nperseg=256, window="hann", noverlap=128)
    ax[0].semilogy(f, P, color=c, label=lab, lw=1.1)
    band[k] = {"low": float(np.trapezoid(P[f < 0.3], f[f < 0.3])), "high": float(np.trapezoid(P[f > 1.0], f[f > 1.0]))}
ax[0].set(xlabel="Frequency, Hz", ylabel="PSD, (m/s)$^2$/Hz", xlim=(0, 10))
ax[0].legend(fontsize=6)
t = np.arange(len(s["w"])) / FS
sl = (t > 100) & (t < 160)
ax[1].plot(t[sl], s["A"][sl], color="0.7", lw=0.5, label="A")
ax[1].plot(t[sl], s["w"][sl], "k", lw=1.2, label="Truth")
ax[1].plot(t[sl], s["B"][sl], color="tab:green", lw=1, label="B")
ax[1].plot(t[sl], s["C"][sl], color="tab:blue", lw=1, label="C")
ax[1].set(xlabel="Time, s", ylabel="Vertical air velocity, m/s", ylim=(-3, 6))
ax[1].legend(fontsize=6, ncol=4, loc="upper right")
fig.tight_layout()
fig.savefig(F / "spectra.pdf")
fig.savefig(F / "spectra.png")
print("\nBAND POWER (m/s)^2:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in band.items()})
summary["band"] = band
json.dump(summary, open(R / "dsp_summary.json", "w"), indent=1, default=float)
print("saved figures")
