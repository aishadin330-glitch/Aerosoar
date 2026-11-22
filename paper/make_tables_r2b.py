"""Second part of the table generation: delay-compensated assignment and replay of recorded flights.
usage (project root): .venv/Scripts/python.exe paper/make_tables_r2b.py       -> paper/tables_r2/{lagcomp,realreplay}.tex"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_rev2 as A

T = ROOT / "tables_r2"
PM = "$\\pm$"
pm = lambda mc, nd=3: f"{mc[0]:.{nd}f}{PM}{mc[1]:.{nd}f}"
keys = [f"{e}|{c}" for e in ("pf", "ekf") for c in ("calm", "tailwind", "adverse")]

# delay-compensated assignment: centre error as estimated -> with compensation
cl = A.closed("")
body = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Particle filter} & \\multicolumn{3}{c}{Extended Kalman filter} \\\\\nCutoff & calm & tailwind & adverse & calm & tailwind & adverse \\\\\n\\midrule\n"
for fc in (0.1, 0.2, 0.4, 0.8, 1.5):
    cells = []
    for k in keys:
        u = cl.get(k, {}).get(f"B fc={fc}")
        c = cl.get(k, {}).get(f"B fc={fc} lag-compensated")
        cells.append(f"{u['ctr'][0]:.1f} $\\to$ {c['ctr'][0]:.1f}" if u and c else "--")
    body += f"{fc:g}~Hz & " + " & ".join(cells) + " \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
(T / "lagcomp.tex").write_text(body, encoding="utf-8")

# replay of recorded flights against the GNSS-based reference
rr = json.load(open(ROOT / "results" / "real_replay_summary.json"))
body = "\\begin{tabular}{@{}lcc@{}}\n\\toprule\nPipeline & quiet & active \\\\\n\\midrule\n"
for p, lab in (("B", "B~Butterworth"), ("C", "C~Kalman"), ("CL", "CL~Kalman, lag-matched"), ("CS", "CS~Kalman, scheduled"), ("CI", "CI~Kalman, IMU-aided speed rate"), ("D", "D~Energy-rate Kalman"), ("E", "E~Complementary")):
    v = rr["pipes"][p]
    body += f"{lab} & {pm(v['steady']['per_segment'], 2)} & {pm(v['man']['per_segment'], 2)} \\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
(T / "realreplay.tex").write_text(body, encoding="utf-8")
print("wrote lagcomp, realreplay")
