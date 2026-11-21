"""The figures of the paper, in one consistent IEEE style.   usage (project root): .venv/Scripts/python.exe paper/make_figures.py

Reads paper/results/dsp.json and rev_summary.json; writes paper/figures/{diagram,spectra,filters,roc}.{pdf,png}
Sizes are the final printed sizes (7.16 in two-column, 3.5 in one-column), so the 8 pt fonts are 8 pt in the paper.
"""
import csv, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import signal

ROOT = pathlib.Path(__file__).resolve().parent
F = ROOT / "figures"
F.mkdir(exist_ok=True)
D = json.load(open(ROOT / "results" / "dsp.json"))
FS = 20.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.5, "grid.linestyle": ":",
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.7", "legend.fancybox": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
# one colour and one line style per signal, readable in grey scale
STYLE = {
    "truth": dict(color="black", ls="-", lw=1.4, label="Truth"),
    "A": dict(color="0.55", ls=":", lw=1.1, label="A: unfiltered"),
    "B": dict(color="#D55E00", ls="--", lw=1.1, label="B: Butterworth, 0.4 Hz"),
    "C": dict(color="#0072B2", ls="-", lw=1.1, label="C: Kalman"),
}
FILT = {
    "butter": dict(color="#0072B2", ls="-", marker="o", label="Butterworth (2nd order)"),
    "movavg": dict(color="#D55E00", ls="--", marker="s", label="Moving average"),
    "ema": dict(color="#009E73", ls="-.", marker="^", label="One-pole"),
}


def caption_below(ax, text, y=-0.235):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def save(fig, name):
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(F / f"{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ spectra
s = {k: np.array(v) for k, v in D["series"].items()}
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.75), gridspec_kw={"wspace": 0.28})
a = ax[0]
a.axvspan(0.07, 0.3, color="#cfe3f1", alpha=0.6, lw=0, zorder=0)
a.text(0.145, 2.2e1, "thermal band", ha="center", va="top", fontsize=6.5, color="#2b5d80")
for k, key in [("w", "truth"), ("A", "A"), ("B", "B"), ("C", "C")]:
    f, P = signal.welch(s[k][200:], fs=FS, nperseg=256, window="hann", noverlap=128)
    a.loglog(f[1:], P[1:], **STYLE[key])
a.axvline(0.4, color="0.3", lw=0.7, ls="-.")
a.text(0.44, 2.2e1, "$f_c=0.4$ Hz", fontsize=6.5, va="top", color="0.25")
a.set(xlabel="Frequency (Hz)", ylabel="PSD ((m/s)$^2$/Hz)", xlim=(0.07, 10), ylim=(1e-8, 3e1))
a.legend(loc="lower left", ncol=1, borderpad=0.4, labelspacing=0.3, handlelength=2.2, bbox_to_anchor=(0.0, 0.0))
caption_below(a, "(a) Power spectral density (Welch)")

b = ax[1]
t = np.arange(len(s["w"])) / FS
sl = (t >= 100) & (t <= 160)
b.plot(t[sl], s["A"][sl], color="0.82", lw=0.5, zorder=1, label="A: unfiltered")
b.plot(t[sl], s["B"][sl], **{**STYLE["B"], "zorder": 3})
b.plot(t[sl], s["C"][sl], **{**STYLE["C"], "zorder": 4})
b.plot(t[sl], s["w"][sl], **{**STYLE["truth"], "zorder": 5})
b.set(xlabel="Time (s)", ylabel="Vertical air velocity (m/s)", xlim=(100, 160), ylim=(-2, 5))
h, l = b.get_legend_handles_labels()
order = [3, 0, 1, 2]
b.legend([h[i] for i in order], [l[i] for i in order], loc="upper right", ncol=2, borderpad=0.4, labelspacing=0.3, columnspacing=1.0, handlelength=2.0)
caption_below(b, "(b) Time-domain excerpt")
save(fig, "spectra")

# ------------------------------------------------------------------ filters
def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


FC = [0.1, 0.2, 0.4, 0.8, 1.5]
fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.75), gridspec_kw={"wspace": 0.28})
a = ax[0]
for t_, st in FILT.items():
    xs, ys, es = [], [], []
    for fc in FC:
        r = [x for x in D["filters"] if x["type"] == t_ and x["fc"] == fc]
        xs.append(np.mean([x["delay"] for x in r]))
        ys.append(np.mean([x["rmse"] for x in r]))
        es.append(ci([x["rmse"] for x in r]))
    a.errorbar(xs, ys, yerr=es, capsize=1.5, ms=3.5, elinewidth=0.6, **st)
    if t_ == "butter":
        for fc, x, y in zip(FC, xs, ys):
            lx, ly = {0.1: (2.2, 0.53), 0.2: (1.3, 0.14), 0.4: (0.85, 0.12), 0.8: (0.62, 0.60), 1.5: (0.55, 0.86)}[fc]
            a.annotate(f"{fc:g} Hz", (x, y), xytext=(lx, ly), fontsize=6.3, color="#0072B2", ha="center", va="center",
                       arrowprops=dict(arrowstyle="-", lw=0.4, color="#0072B2", shrinkA=0, shrinkB=2))
a.set(xlabel="Measured delay (s)", ylabel="RMSE of vertical air velocity (m/s)", xlim=(0, 2.6), ylim=(0, 1.35))
a.legend(loc="upper right", title="Filter (cutoff $f_c$ annotated)", title_fontsize=6.5, borderpad=0.4, labelspacing=0.3)
caption_below(a, "(a) Error against delay, 30 flights per point")

b = ax[1]
fc = 0.4
K = np.tan(np.pi * fc / FS)
n = 1 / (1 + np.sqrt(2) * K + K * K)
bb = np.array([K * K * n, 2 * K * K * n, K * K * n])
aa = np.array([1, 2 * (K * K - 1) * n, (1 - np.sqrt(2) * K + K * K) * n])
N = round(0.443 * FS / fc)
al = 1 - np.exp(-2 * np.pi * fc / FS)
for t_, (num, den) in {"butter": (bb, aa), "movavg": (np.ones(N) / N, [1.0]), "ema": ([al], [1, -(1 - al)])}.items():
    f, H = signal.freqz(num, den, worN=2048, fs=FS)
    st = {k: v for k, v in FILT[t_].items() if k != "marker"}
    b.plot(f, 20 * np.log10(np.maximum(np.abs(H), 1e-4)), **st)
b.axvline(fc, color="0.3", lw=0.7, ls="-.")
b.axhline(-3, color="0.3", lw=0.7, ls=":")
b.text(0.46, -47, "$f_c=0.4$ Hz", fontsize=6.5, color="0.25")
b.text(2.95, -2.2, "$-3$ dB", fontsize=6.5, color="0.25", ha="right")
b.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", xlim=(0, 3), ylim=(-50, 2))
caption_below(b, "(b) Magnitude response at $f_c=0.4$ Hz (line styles as in (a))")
save(fig, "filters")

# ------------------------------------------------------------------ detector operating curves
RS = json.load(open(ROOT / "results" / "rev_summary.json"))["roc"]
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.55), sharey=True, gridspec_kw={"wspace": 0.08})
fam = {"thr": ("Threshold (level varied)", "#D55E00", "--", "o"), "fix": ("Fixed CUSUM (h varied)", "#0072B2", "-", "s"), "ada": ("Adaptive CUSUM (h varied)", "#009E73", "-.", "^")}
for a, turb, cap in zip(ax, (0.2, 0.5, 0.9), ("(a) turbulence 0.2 m/s", "(b) turbulence 0.5 m/s", "(c) turbulence 0.9 m/s")):
    for f, (lab, c, ls, m) in fam.items():
        pts = sorted((v["fa"], v["pd"]) for k, v in RS.items() if k.startswith(f"{turb}|test|{f}|"))
        a.plot([p_[0] for p_ in pts], [100 * p_[1] for p_ in pts], color=c, ls=ls, marker=m, ms=3.4, label=lab)
    a.axvline(5, color="0.3", lw=0.7, ls=":")
    a.set_xscale("symlog", linthresh=1, linscale=0.5)
    top = {0.2: 10, 0.5: 100, 0.9: 300}[turb]
    a.set(xlim=(0, top), ylim=(0, 102), xlabel="False alarms per hour")
    tk = {10: [0, 1], 100: [0, 1, 10], 300: [0, 1, 10, 100]}[top]
    a.set_xticks(tk); a.set_xticklabels([str(t) for t in tk])
    caption_below(a, cap, y=-0.30)
ax[0].set_ylabel("Detection rate (%)")
ax[2].legend(loc="lower right", borderpad=0.4, labelspacing=0.3, handlelength=2.2)
save(fig, "roc")

# ------------------------------------------------------------------ system diagram
fig, a = plt.subplots(figsize=(7.16, 2.55))
a.set_xlim(0, 100)
a.set_ylim(0, 36)
a.axis("off")
a.grid(False)


def box(x, y, w, h, text, fc="#ffffff", ec="black", lw=0.8, fs=7.5, weight="normal"):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", fc=fc, ec=ec, lw=lw))
    a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25, weight=weight)


def arrow(x0, y0, x1, y1, text=None, dy=1.2):
    a.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", shrinkA=0, shrinkB=0, mutation_scale=7))
    if text:
        a.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.5, style="italic")


ys = [27.5, 20, 12.5, 5]
for (t, r), y in zip([("IMU", "100 Hz"), ("Barometer", "20 Hz"), ("Pitot tube", "25 Hz"), ("GNSS", "5 Hz, 0.15 s delay")], ys):
    box(0.5, y, 15, 5.6, f"{t}\n{r}", fc="#e8f0f8")
    arrow(16, y + 2.8, 21.5, y + 2.8)
a.text(8, 34.6, "Sensors", ha="center", fontsize=8, weight="bold")

a.add_patch(FancyBboxPatch((21.5, 1.5), 36, 31.5, boxstyle="round,pad=0,rounding_size=1.0", fc="#f1f6ee", ec="black", lw=0.8))
a.text(39.5, 34.6, "Causal multi-rate signal processing", ha="center", fontsize=8, weight="bold")
box(23.5, 22.5, 32, 7.5, "Vertical Kalman filter\n$[h,\\;v_z,\\;b_a,\\;b_b]$", fc="#ffffff")
box(23.5, 13, 32, 7.5, "Airspeed filter $[V,\\;\\dot V]$,\nnavigation and wind", fc="#ffffff")
box(23.5, 3.5, 32, 7.5, "Energy equation\n$w=\\dot h+V\\dot V/g+$ sink rate", fc="#ffffff")
arrow(39.5, 22.5, 39.5, 20.5)
arrow(39.5, 13, 39.5, 11)

box(62, 20, 16.5, 9, "Vertical air\nvelocity $\\hat w,\\;\\sigma_w$\n(20 Hz)", fc="#fff3d6", fs=7)
arrow(57.5, 24.5, 62, 24.5)
box(62, 4.5, 16.5, 9, "Adaptive CUSUM\ndetector", fc="#fff3d6")
arrow(70.25, 20, 70.25, 13.5)
box(84, 20, 16, 9, "Thermal estimator\n(PF, EKF)", fc="#f9e4e4")
arrow(78.5, 24.5, 84, 24.5)
box(84, 4.5, 16, 9, "Guidance\ncruise or climb", fc="#f9e4e4")
arrow(78.5, 9, 84, 9)
a.text(81.25, 9.9, "alarm", ha="center", va="bottom", fontsize=6.3, style="italic")
arrow(92, 20, 92, 13.5)
a.text(93, 16.7, "centre", ha="left", va="center", fontsize=6.3, style="italic")
save(fig, "diagram")
