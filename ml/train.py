"""Train and compare an LSTM and a Transformer on simulator windows.

usage: train.py <lstm|transformer> [epochs]
Writes ml/model/<name>.json (weights, calibration, test vectors) and ml/model/<name>_report.json.
"""
import glob, json, sys, time, pathlib
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(4)
WIN, NF = 32, 9
ROOT = pathlib.Path(__file__).resolve().parent.parent  # project folder, so scripts run from anywhere


def load(split):
    xs = sorted(glob.glob(str(ROOT / "ml" / "data" / f"{split}_*.x.f32")))
    X = np.concatenate([np.fromfile(f, np.float32) for f in xs]).reshape(-1, WIN, NF)
    Y = np.concatenate([np.fromfile(f.replace(".x.", ".y."), np.float32) for f in xs]).reshape(-1, 6)
    return torch.from_numpy(X), torch.from_numpy(Y)


class LSTMNet(nn.Module):
    def __init__(s, h=64):
        super().__init__()
        s.rnn = nn.LSTM(NF, h, num_layers=2, batch_first=True)
        s.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 9))

    def forward(s, x):
        o, _ = s.rnn(x)
        return s.head(o[:, -1])


class TransformerNet(nn.Module):
    def __init__(s, d=64, heads=4, layers=2):
        super().__init__()
        s.inp = nn.Linear(NF, d)
        s.pos = nn.Parameter(torch.randn(WIN, d) * 0.02)
        s.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 128, dropout=0.0, batch_first=True, norm_first=True), layers)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 9))

    def forward(s, x):
        h = s.enc(s.inp(x) + s.pos)
        return s.head(s.norm(h[:, -1]))  # the newest sample summarises the window


def loss_fn(out, y):
    logit, mu, lv = out[:, 0], out[:, 1:5], out[:, 5:9].clamp(-6, 3)
    ex = y[:, 0]
    bce = nn.functional.binary_cross_entropy_with_logits(logit, ex)
    nll = 0.5 * (lv + (y[:, 1:5] - mu) ** 2 / lv.exp())
    near = 1.0 + 2.0 * (torch.hypot(y[:, 1], y[:, 2]) < 1.2).float()  # windows within 120 m count triple
    reg = (nll.sum(1) * ex * near).sum() / (ex * near).sum().clamp(min=1)
    return bce + reg


@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def metrics(out, Y, k=1.0):
    pos = Y[:, 0] == 1
    p = torch.sigmoid(out[:, 0])
    acc = ((p > 0.5).float() == Y[:, 0]).float().mean().item()
    o, y = out[pos], Y[pos]
    err = torch.hypot(o[:, 1] - y[:, 1], o[:, 2] - y[:, 2]) * 100
    var = o[:, 5:9].clamp(-6, 3).exp() * k
    d2 = (o[:, 1] - y[:, 1]) ** 2 / var[:, 0] + (o[:, 2] - y[:, 2]) ** 2 / var[:, 1]
    return dict(
        exist_acc=acc,
        centre_err_m=err.mean().item(),
        centre_err_median_m=err.median().item(),
        near_err_m=err[torch.hypot(y[:, 1], y[:, 2]) < 1.2].mean().item(),
        w_mae=((o[:, 3] - y[:, 3]).abs().mean() * 4).item(),
        sigma_mae_m=((o[:, 4] - y[:, 4]).abs().mean() * 60).item(),
        coverage95=(d2 < 5.991).float().mean().item(),
        d2=d2,
    )


def main():
    name, epochs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
    model = LSTMNet() if name == "lstm" else TransformerNet()
    nparam = sum(p.numel() for p in model.parameters())
    Xtr, Ytr = load("train"); Xva, Yva = load("val"); Xte, Yte = load("test"); Xod, Yod = load("ood")
    print(name, "params", nparam, "train", len(Xtr), flush=True)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    steps = epochs * (len(Xtr) // 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    t0, best, best_state = time.time(), 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm) - 511, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        m = metrics(predict(model, Xva), Yva)
        vl = loss_fn(predict(model, Xva), Yva).item()
        print(f"epoch {ep+1}/{epochs} val_loss {vl:.3f} centre_err {m['centre_err_m']:.1f} m cov95 {m['coverage95']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # calibration: scale predicted variances so that 95 % of validation positives fall inside the 95 % region
    mv = metrics(predict(model, Xva), Yva)
    k = float(np.quantile(mv["d2"].numpy(), 0.95) / 5.991)
    rep = {"model": name, "params": nparam, "epochs": epochs, "train_windows": len(Xtr), "var_scale": k}
    for tag, X, Y in [("test", Xte, Yte), ("ood", Xod, Yod)]:
        m = metrics(predict(model, X), Y, k)
        m.pop("d2")
        rep[tag] = m
    one = Xte[:1]
    model.eval()
    t = time.time()
    with torch.no_grad():
        for _ in range(200):
            model(one)
    rep["infer_ms"] = (time.time() - t) / 200 * 1000
    print(json.dumps(rep, indent=1), flush=True)

    # export weights as nested lists, plus test vectors for checking the JavaScript forward pass
    sd = {k_: v.numpy().round(6).tolist() for k_, v in model.state_dict().items()}
    with torch.no_grad():
        tv = predict(model, Xte[:5])
    json.dump({"name": name, "win": WIN, "nf": NF, "var_scale": k, "weights": sd,
               "test_x": Xte[:5].numpy().tolist(), "test_y": tv.numpy().round(6).tolist()},
              open(ROOT / "ml" / "model" / f"{name}.json", "w"), separators=(",", ":"))
    json.dump(rep, open(ROOT / "ml" / "model" / f"{name}_report.json", "w"), indent=1)



if __name__ == "__main__":
    main()
