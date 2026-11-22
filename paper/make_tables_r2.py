"""LaTeX tables of the second revision, generated from the result files so that no number is typed by hand.
usage (project root): .venv/Scripts/python.exe paper/make_tables_r2.py       -> paper/tables_r2/*.tex"""
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_rev2 as A

T = ROOT / "tables_r2"
T.mkdir(exist_ok=True)
PM = "$\\pm$"


def pm(mc, nd=3):
    return f"{mc[0]:.{nd}f}{PM}{mc[1]:.{nd}f}"


def write(name, body):
    (T / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


LAB = {"A": "A~Unfiltered", "B": "B~Butterworth", "C": "C~Kalman", "CL": "CL~Kalman, lag-matched", "C5S": "CS~Kalman, scheduled",
       "D": "D~Energy-rate Kalman", "D2": "D2~Energy-rate Kalman + acc.", "E": "E~Complementary", "CI": "CI~Kalman, IMU-aided speed rate", "Cx": "Cx~Kalman, true speed rate"}

# ---------------------------------------------------------------- offline filter baselines
rows = A.baselines()
want = ["Butterworth 0.4 Hz", "Bessel 0.4 Hz", "Moving average 0.4 Hz", "One-pole 0.4 Hz", "Butterworth + lead compensation (0.2 s)",
        "Savitzky-Golay N=61, p=2, look-ahead 0.00 s", "Savitzky-Golay N=41, p=2, look-ahead 0.75 s", "Savitzky-Golay N=61, p=3, look-ahead 0.50 s"]
nice = {"Butterworth 0.4 Hz": "Butterworth, 0.4 Hz", "Bessel 0.4 Hz": "Bessel, 0.4 Hz", "Moving average 0.4 Hz": "Moving average, 0.4 Hz", "One-pole 0.4 Hz": "One-pole, 0.4 Hz",
        "Butterworth + lead compensation (0.2 s)": "Butterworth + lead compensation (0.2 s)",
        "Savitzky-Golay N=61, p=2, look-ahead 0.00 s": "Causal Savitzky--Golay, 3 s, $p=2$",
        "Savitzky-Golay N=41, p=2, look-ahead 0.75 s": "Asymmetric Savitzky--Golay, 2 s, $p=2$",
        "Savitzky-Golay N=61, p=3, look-ahead 0.50 s": "Asymmetric Savitzky--Golay, 3 s, $p=3$"}
d = {n: (dl, mc) for n, dl, mc in rows}
body = "\\begin{tabular}{@{}lcc@{}}\n\\toprule\nFilter & Delay (s) & RMSE (m/s) \\\\\n\\midrule\n"
for n in want:
    dl, mc = d[n]
    best = n.startswith("Butterworth 0.4")
    body += f"{nice[n]} & {dl:.2f} & {pm(mc)} \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("baselines", body)

# ---------------------------------------------------------------- pipelines, three sensor settings
res = {s: A.pipelines(s) for s in ("", "_real", "_trace")}
order = ["A", "B", "C", "CL", "C5S", "CI", "D", "D2", "E", "Cx"]
body = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Nominal sensor model} & \\multicolumn{2}{c}{Recorded rates, noise level} & \\multicolumn{2}{c}{Recorded noise replayed} \\\\\n"
body += "Pipeline & steady & speed change & steady & speed change & steady & speed change \\\\\n\\midrule\n"
for p in order:
    if p == "Cx":
        body += "\\midrule\n"
    cells = []
    for s in ("", "_real", "_trace"):
        x = res[s]["nominal"][p]
        nd = 2 if p == "A" else 3
        cells += [pm(x["steady"], nd), pm(x["man"], nd)]
    body += f"{LAB[p]} & " + " & ".join(cells) + " \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("pipelines", body)

# ---------------------------------------------------------------- faults and model errors (nominal sensors)
conds = [("drift", "Barometer drift ramp"), ("pitot", "Pitot scale error 7\\%"), ("polarM", "Drag polar $-20\\%$"), ("polarP", "Drag polar $+20\\%$"),
         ("tilt1", "Attitude error $1^\\circ$"), ("tilt1B", "Attitude error $1^\\circ$, allowance"), ("aoa1", "Accelerometer alignment $1^\\circ$"),
         ("noise3", "Sensor noise $\\times3$")]
cols = ["B", "C", "C5S", "CI", "D", "D2"]
nom = res[""]["nominal"]
body = "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\nCondition & " + " & ".join(LAB[c].split("~")[0] for c in cols) + " \\\\\n\\midrule\n"
body += "Nominal & " + " & ".join(f"{nom[c]['steady'][0]:.3f} ({nom[c]['bias'][0]:+.2f})" for c in cols) + " \\\\\n"
for key, nm in conds:
    r = res[""][key]
    body += f"{nm} & " + " & ".join(f"{r[c]['steady'][0]:.3f} ({r[c]['bias'][0]:+.2f})" for c in cols) + " \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("conditions", body)

# ---------------------------------------------------------------- detectors at matched false-alarm limits
dk = A.detect("kf", 0)
body = "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n & & \\multicolumn{2}{c}{$\\le5$ false alarms per hour} & \\multicolumn{2}{c}{$\\le2$ false alarms per hour} \\\\\n"
body += "Turbulence & Detector & detection (\\%) & false alarms/h & detection (\\%) & false alarms/h \\\\\n\\midrule\n"
NM = {"thr": "Threshold", "fix": "Fixed CUSUM", "ada": "Adaptive CUSUM", "glr": "Matched filter"}
for turb in (0.2, 0.5, 0.9):
    fl = [f for f in ("thr", "fix", "ada") if f in dk[f"{turb}|5"]]  # the matched-filter family of the result files is reported in the text only
    for i, f in enumerate(fl):
        cells = []
        for lim in (5, 2):
            x = dk[f"{turb}|{lim}"][f]
            if x.get("key") is None:
                cells += ["\\multicolumn{2}{c}{none meets the limit}"]
            else:
                cells += [f"{100 * x['pd'][0]:.1f} [{100 * x['pd'][1]:.1f}, {100 * x['pd'][2]:.1f}]", f"{x['fa'][0]:.1f} [{x['fa'][1]:.1f}, {x['fa'][2]:.1f}]"]
        body += (f"{turb:g}~m/s" if i == 0 else "") + f" & {NM[f]} & " + " & ".join(cells) + " \\\\\n"
    for key, lab in (("ada-thr", "adaptive $-$ threshold"),):
        if key not in dk[f"{turb}|5"]:
            continue
        cells = []
        for lim in (5, 2):
            x = dk[f"{turb}|{lim}"].get(key)
            cells += [f"\\multicolumn{{2}}{{c}}{{{100 * x[0]:+.1f} [{100 * x[1][0]:+.1f}, {100 * x[1][1]:+.1f}] points}}" if x else "\\multicolumn{2}{c}{--}"]
        body += f" & \\textit{{{lab}}} & " + " & ".join(cells) + " \\\\\n"
    if turb != 0.9:
        body += "\\midrule\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("detect", body)

# ---------------------------------------------------------------- detection while the speed changes
body = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Threshold} & \\multicolumn{2}{c}{Fixed CUSUM} & \\multicolumn{2}{c}{Adaptive CUSUM} \\\\\nSignal source & det. (\\%) & FA/h & det. (\\%) & FA/h & det. (\\%) & FA/h \\\\\n\\midrule\n"
for src, nm in (("kf", "C~Kalman"), ("dsp", "B~Butterworth"), ("kfl", "CL~Kalman, lag-matched"), ("kf5s", "CS~Kalman, scheduled"), ("en", "D~Energy-rate Kalman")):
    r = A.detect(src, 2)
    if not r:
        continue
    row = r["0.5|5"]
    cells = []
    for f in ("thr", "fix", "ada"):
        x = row[f]
        cells += [f"{100 * x['pd'][0]:.1f}", f"{x['fa'][0]:.1f}"] if x.get("key") else ["--", "--"]
    body += f"{nm} & " + " & ".join(cells) + " \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("doublet", body)

# ---------------------------------------------------------------- false-alarm interval
arl = A.arl()
fmt = lambda v: "none" if not np.isfinite(v) else (f"{v:.0f}" if v < 1e4 else f"{v / 1e3:.0f}k")
body = "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n & & \\multicolumn{3}{c}{Fixed CUSUM} & \\multicolumn{2}{c}{Adaptive CUSUM} \\\\\nTurbulence & $h_0$ & simulator & AR(1) & i.i.d. & simulator & AR(1) \\\\\n\\midrule\n"
for turb in (0.5, 0.9):
    st = [x for x in arl if x["turb"] == turb][0]
    for i, h in enumerate((1.5, 3.0, 6.0)):
        fx = [x for x in arl if x["turb"] == turb and x["h0"] == h and not x["ada"]][0]
        ad = [x for x in arl if x["turb"] == turb and x["h0"] == h and x["ada"]][0]
        lead = f"{turb:g}~m/s ($\\phi={st['phi']:.3f}$)" if i == 0 else ""
        body += f"{lead} & {h:g} & {fmt(fx['real'])} & {fmt(fx['ar1'])} & {fmt(fx['iid'])} & {fmt(ad['real'])} & {fmt(ad['ar1'])} \\\\\n"
    if turb == 0.5:
        body += "\\midrule\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("arl", body)

# ---------------------------------------------------------------- closed loop: centre error per estimator and condition
cl = A.closed("")
names = [("B fc=0.4", "B~Butterworth, 0.4~Hz"), ("B fc=1.5", "B~Butterworth, 1.5~Hz"), ("C kalman", "C~Kalman"), ("C+lag Kalman", "CL~Kalman, lag-matched"),
         ("C5S scheduled Kalman", "CS~Kalman, scheduled"), ("CI IMU-aided speed-rate Kalman", "CI~Kalman, IMU-aided speed rate"), ("D energy-rate Kalman", "D~Energy-rate Kalman"),
         ("D2 energy-rate Kalman + accelerometer", "D2~Energy-rate Kalman + acc."), ("E complementary", "E~Complementary")]
keys = [f"{e}|{c}" for e in ("pf", "ekf") for c in ("calm", "tailwind", "adverse")]
body = "\\begin{tabular}{@{}l" + "c" * 6 + "@{}}\n\\toprule\n & \\multicolumn{3}{c}{Particle filter} & \\multicolumn{3}{c}{Extended Kalman filter} \\\\\nPipeline & calm & tailwind & adverse & calm & tailwind & adverse \\\\\n\\midrule\n"
for nm, lab in names:
    cells = []
    for k in keys:
        x = cl.get(k, {}).get(nm)
        cells.append(f"{x['ctr'][0]:.1f}{PM}{x['ctr'][1]:.1f}" if x else "--")
    body += f"{lab} & " + " & ".join(cells) + " \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("closed", body)
print("speed range over all closed-loop configurations:", min(x["speed"][0] for k in cl.values() for x in k.values()), max(x["speed"][0] for k in cl.values() for x in k.values()))
