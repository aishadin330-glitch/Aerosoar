"""Show what a trained model answers for individual test windows (answer vs. truth).

usage: uv run --no-project --python 3.12 --with torch --with numpy python -I ml/predict_demo.py [lstm|transformer] [count]
Run from the project root. Loads the exported weights, so no retraining is needed.
"""
import json, sys, pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train import LSTMNet, TransformerNet, load  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "transformer"
count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
j = json.load(open(pathlib.Path(__file__).resolve().parent / "model" / f"{name}.json"))
model = LSTMNet() if name == "lstm" else TransformerNet()
model.load_state_dict({k: torch.tensor(v) for k, v in j["weights"].items()})
model.eval()

X, Y = load("test")
rng = np.random.default_rng(0)
idx = rng.choice(len(X), count, replace=False)
with torch.no_grad():
    out = model(X[idx])
p = torch.sigmoid(out[:, 0])
sd = (out[:, 5:9].clamp(-6, 3).exp() * j["var_scale"]).sqrt()
print(f"{name}: {count} random windows from unseen test flights\n")
print(f"{'#':>2} {'thermal?':>8} {'P(thermal)':>10} | {'true centre (E,N) m':>20} {'predicted centre m':>20} {'error m':>8} | {'true w':>6} {'pred w':>6} | {'true sigma':>6} {'pred sigma':>6}")
for i, k in enumerate(idx):
    y, o = Y[k], out[i]
    real = bool(y[0])
    if real:
        tc = (y[1] * 100, y[2] * 100)
        pc = (o[1] * 100, o[2] * 100)
        err = float(torch.hypot(tc[0] - pc[0], tc[1] - pc[1]))
        print(f"{i:>2} {'yes':>8} {p[i]:>10.2f} | ({tc[0]:>6.0f},{tc[1]:>6.0f}) m ({pc[0]:>6.0f},{pc[1]:>6.0f}) +-{2*sd[i,0]*100:>4.0f} {err:>8.0f} | {y[3]*4:>6.1f} {o[3]*4:>6.1f} | {y[4]*60:>6.0f} {o[4]*60:>6.0f}")
    else:
        print(f"{i:>2} {'no':>8} {p[i]:>10.2f} | no thermal within 250 m with useful lift; regression outputs are ignored")
