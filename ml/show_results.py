"""Show the results of model training: tables from the reports and learning curves from the logs.

usage: uv run --no-project --python 3.12 --with matplotlib python -I ml/show_results.py
Writes docs/figures/ml_training_curves.png.
"""
import json, re, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parent
out = root.parent / "docs" / "figures"
out.mkdir(parents=True, exist_ok=True)
names = {"lstm": "LSTM", "transformer": "Transformer"}

print(f"{'Model':<12}{'Params':>8}  {'Test: near-thermal err':>24}{'all-window err':>16}{'coverage95':>12}{'existence acc':>15}   OOD near err / coverage")
curves = {}
for key, label in names.items():
    r = json.load(open(root / "model" / f"{key}_report.json"))
    t, o = r["test"], r["ood"]
    print(f"{label:<12}{r['params']:>8}  {t['near_err_m']:>22.1f} m{t['centre_err_m']:>14.1f} m{t['coverage95']:>12.2f}{t['exist_acc']:>15.2f}   {o['near_err_m']:.1f} m / {o['coverage95']:.2f}")
    rows = re.findall(r"epoch (\d+)/\d+ val_loss (-?[\d.]+) centre_err ([\d.]+) m cov95 ([\d.]+)", open(root / "model" / f"{key}.log").read())
    curves[label] = [(int(e), float(l), float(c)) for e, l, c, _ in rows]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for label, c in curves.items():
    ax[0].plot([x[0] for x in c], [x[1] for x in c], marker="o", label=label)
    ax[1].plot([x[0] for x in c], [x[2] for x in c], marker="o", label=label)
ax[0].set(xlabel="Epoch", ylabel="Validation loss", title="Validation loss (lower is better)")
ax[1].set(xlabel="Epoch", ylabel="Centre error, m", title="Validation centre error, all windows")
for a in ax:
    a.grid(alpha=0.3)
    a.legend()
fig.tight_layout()
fig.savefig(out / "ml_training_curves.png", dpi=150)
print("saved", out / "ml_training_curves.png")
