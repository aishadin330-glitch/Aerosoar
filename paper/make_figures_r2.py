"""Figures of the second revision (new file names; the figures of the earlier paper are not touched).
usage (project root): .venv/Scripts/python.exe paper/make_figures_r2.py
Reads paper/results/rev2_*.json; writes paper/figures/{roc2,pipelines2,ctrfit}.{pdf,png}"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_rev2 as A

F = ROOT / "figures"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.5, "grid.linestyle": ":",
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.7", "legend.fancybox": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(F / f"{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    print("saved", name)


def caption_below(ax, text, y=-0.30):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=8)


# ------------------------------------------------------------------ detector operating curves (finer grids)
data = A.load("rev2_detect_kf_0")
fam = {"thr": ("Threshold (level varied)", "#D55E00", "--", "o"), "fix": ("Fixed CUSUM ($h_0$ varied)", "#0072B2", "-", "s"), "ada": ("Adaptive CUSUM ($h_0$ varied)", "#009E73", "-.", "^")}
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True, gridspec_kw={"wspace": 0.08})
for a, turb, cap in zip(ax, (0.2, 0.5, 0.9), ("(a) turbulence 0.2 m/s", "(b) turbulence 0.5 m/s", "(c) turbulence 0.9 m/s")):
    rs = [r for r in data if r["split"] == "test" and r["turb"] == turb]
    for f, (lab, c, ls, m) in fam.items():
        pts = sorted(A.rate(rs, k)[::-1] for k in data[0]["det"] if k.startswith(f + "|"))
        a.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=c, ls=ls, marker=m, ms=3.4, label=lab)
    a.axvline(5, color="0.3", lw=0.7, ls=":")
    top = {0.2: 10, 0.5: 100, 0.9: 300}[turb]
    a.set_xscale("symlog", linthresh=1, linscale=0.5)
    a.set(xlim=(0, top), ylim=(0, 102), xlabel="False alarms per hour")
    tk = {10: [0, 1], 100: [0, 1, 10], 300: [0, 1, 10, 100]}[top]
    a.set_xticks(tk); a.set_xticklabels([str(t) for t in tk])
    caption_below(a, cap)
ax[0].set_ylabel("Detection rate (%)")
ax[2].legend(loc="lower right", borderpad=0.4, labelspacing=0.3, handlelength=2.2)
save(fig, "roc2")

# ------------------------------------------------------------------ pipelines: steady against speed-change error
STY = {"B": ("B  Butterworth", "#D55E00", "o"), "C": ("C  Kalman", "#0072B2", "s"), "CL": ("CL  lag-matched Kalman", "#56B4E9", "D"),
       "C5S": ("CS  scheduled Kalman", "#000000", "*"), "D": ("D  energy-rate Kalman", "#009E73", "^"), "D2": ("D2  + accelerometer drag", "#CC79A7", "v"),
       "E": ("E  complementary", "#E69F00", "P")}
res = {s: A.pipelines(s) for s in ("", "_real", "_trace")}
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6), sharey=True, gridspec_kw={"wspace": 0.08})
for a, s, cap in zip(ax, ("", "_real", "_trace"), ("(a) nominal sensor model", "(b) recorded rates and noise level", "(c) recorded noise replayed")):
    d = res[s]
    if not d:
        continue
    d = d["nominal"]
    for p, (lab, c, m) in STY.items():
        if p not in d:
            continue
        x, ex = d[p]["steady"]; y, ey = d[p]["man"]
        a.errorbar(x, y, xerr=ex, yerr=ey, fmt=m, color=c, ms=5 if m != "*" else 8, elinewidth=0.6, capsize=1.2, label=lab if s == "" else None)
    a.set(xlim=(0.12, 0.40), ylim=(0.2, 0.82), xlabel="Steady-flight RMSE (m/s)")
    caption_below(a, cap)
ax[0].set_ylabel("Speed-change RMSE (m/s)")
ax[0].legend(loc="upper right", borderpad=0.4, labelspacing=0.25, handletextpad=0.3, fontsize=6.3)
save(fig, "pipelines2")

# ------------------------------------------------------------------ latency model of the centre error
c = A.closed("")
fit = A.latency_fit(c)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
fcs = np.array([0.1, 0.2, 0.4, 0.8, 1.5])
tau = np.sqrt(2) / (2 * np.pi * fcs)
colr = {"pf|calm": "#0072B2", "pf|tailwind": "#D55E00", "pf|adverse": "#009E73"}
lab = {"pf|calm": "calm", "pf|tailwind": "tailwind", "pf|adverse": "adverse"}
for key in ("pf|calm", "pf|tailwind", "pf|adverse"):
    if key not in c:
        continue
    y = [c[key][f"B fc={f}"]["ctr"][0] for f in fcs]; e = [c[key][f"B fc={f}"]["ctr"][1] for f in fcs]
    ax.errorbar(tau, y, yerr=e, fmt="o", ms=3.5, color=colr[key], capsize=1.5, elinewidth=0.6, label=f"{lab[key]}")
    t = np.linspace(0.08, 2.4, 100)
    f_ = fit[key]
    ax.plot(t, np.sqrt(f_["s0"] ** 2 + (f_["k"] * 10.5 * t) ** 2), color=colr[key], lw=0.9)
    if all(f"B fc={f} lag-compensated" in c[key] for f in fcs):
        yc = [c[key][f"B fc={f} lag-compensated"]["ctr"][0] for f in fcs]; ec = [c[key][f"B fc={f} lag-compensated"]["ctr"][1] for f in fcs]
        ax.errorbar(tau * 1.04, yc, yerr=ec, fmt="s", ms=3.3, mfc="white", color=colr[key], capsize=1.5, elinewidth=0.6)
ax.set(xlabel="Butterworth group delay $\\tau$ (s)", ylabel="Thermal-centre error (m)", xscale="log", xlim=(0.08, 2.6), ylim=(8, 90))
ax.set_xticks([0.1, 0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"])
ax.legend(loc="upper center", ncol=3, title="particle filter, 30 flights", title_fontsize=6.5, borderpad=0.4, columnspacing=1.2, handletextpad=0.3)
save(fig, "ctrfit")
