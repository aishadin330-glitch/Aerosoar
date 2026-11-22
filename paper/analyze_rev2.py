"""Analysis of the second-revision experiments (gui/tests/paper_rev2.mjs and the offline filter baselines).
usage (project root): .venv/Scripts/python.exe paper/analyze_rev2.py [section ...]   sections: pipelines detect variants arl closed baselines
Writes paper/results/rev2_summary.json. Intervals are Student-t over flights (n = 30) unless stated; rates use a bootstrap over flights."""
import json, sys, pathlib
import numpy as np
from scipy import signal, stats
from collections import defaultdict

R = pathlib.Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(7)
OUT = {}
tq = lambda n: stats.t.ppf(0.975, n - 1)


def load(name):
    p = R / (name + ".json")
    return json.load(open(p)) if p.exists() else None


def mci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(x.mean()), float(tq(len(x)) * x.std(ddof=1) / np.sqrt(len(x)))


# ---------------------------------------------------------------- pipelines
def pipelines(suf=""):
    r = load("rev2_pipelines" + suf)
    if r is None:
        return None
    P = list(r[0]["se"].keys())
    res = {}
    for cond in sorted({x["cond"] for x in r}):
        rs = sorted([x for x in r if x["cond"] == cond], key=lambda x: x["seed"])
        d = {}
        per = {}
        for p in P:
            st = [np.sqrt((x["se"][p]["lvl"] + x["se"][p]["ban"]) / max(1, x["n"]["lvl"] + x["n"]["ban"])) for x in rs]
            ma = [np.sqrt(x["se"][p]["man"] / x["n"]["man"]) if x["n"]["man"] > 0 else np.nan for x in rs]
            bn = [np.sqrt(x["se"][p]["ban"] / x["n"]["ban"]) if x["n"]["ban"] > 0 else np.nan for x in rs]
            per[p] = dict(st=np.array(st), ma=np.array(ma), bn=np.array(bn))
            d[p] = dict(steady=mci(st), man=mci(ma), banked=mci(bn), bias=mci([x["bias"][p] for x in rs]))
        d["_n"] = dict(lvl=float(np.mean([x["n"]["lvl"] for x in rs])), ban=float(np.mean([x["n"]["ban"] for x in rs])), man=float(np.mean([x["n"]["man"] for x in rs])))
        # paired differences to B (per-flight RMSE), Student-t
        diff = {}
        for p in P:
            if p in ("A", "B"):
                continue
            for key in ("st", "ma", "bn"):
                dv = per[p][key] - per["B"][key]
                dv = dv[np.isfinite(dv)]
                diff[f"{p}-B|{key}"] = (float(dv.mean()), float(tq(len(dv)) * dv.std(ddof=1) / np.sqrt(len(dv))))
        for a, b in (("C5", "C"), ("CL", "C"), ("C5L", "CL"), ("D2", "D")):
            for key in ("st", "ma", "bn"):
                dv = per[a][key] - per[b][key]
                dv = dv[np.isfinite(dv)]
                diff[f"{a}-{b}|{key}"] = (float(dv.mean()), float(tq(len(dv)) * dv.std(ddof=1) / np.sqrt(len(dv))))
        d["_diff"] = diff
        res[cond] = d
    return res


# ---------------------------------------------------------------- detectors
def rate(rs, k):
    enc = sum(r["enc"] for r in rs)
    T = sum(r["t"] for r in rs) / 3600
    return sum(r["det"][k]["hit"] for r in rs) / enc, sum(r["det"][k]["fa"] for r in rs) / T


def boot(rs, k, n=2000):
    enc = np.array([r["enc"] for r in rs]); hit = np.array([r["det"][k]["hit"] for r in rs]); fa = np.array([r["det"][k]["fa"] for r in rs]); T = np.array([r["t"] for r in rs]) / 3600
    lat = np.array([np.mean(r["det"][k]["lat"]) if r["det"][k]["lat"] else np.nan for r in rs])
    P, F = [], []
    for _ in range(n):
        i = rng.integers(0, len(rs), len(rs))
        P.append(hit[i].sum() / enc[i].sum()); F.append(fa[i].sum() / T[i].sum())
    return dict(pd=[hit.sum() / enc.sum(), *np.percentile(P, [2.5, 97.5])], fa=[fa.sum() / T.sum(), *np.percentile(F, [2.5, 97.5])], lat=float(np.nanmean(lat)))


def select(data, keys, turb, limit):
    tune = [r for r in data if r["split"] == "tune" and r["turb"] == turb]
    test = [r for r in data if r["split"] == "test" and r["turb"] == turb]
    best = None
    for k in keys:
        pd, fa = rate(tune, k)
        if fa <= limit and (best is None or pd > rate(tune, best)[0]):
            best = k
    return best, (boot(test, best) if best else None)


def paired_pd_diff(data, turb, ka, kb, n=2000):
    test = [r for r in data if r["split"] == "test" and r["turb"] == turb]
    ea = np.array([r["det"][ka]["hit"] for r in test]); eb = np.array([r["det"][kb]["hit"] for r in test]); en = np.array([r["enc"] for r in test])
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(test), len(test))
        ds.append(ea[i].sum() / en[i].sum() - eb[i].sum() / en[i].sum())
    return float((ea.sum() - eb.sum()) / en.sum()), [float(x) for x in np.percentile(ds, [2.5, 97.5])]


def detect(src, dbl, suf=""):
    data = load(f"rev2_detect_{src}_{dbl}{suf}")
    if data is None:
        return None
    keys = list(data[0]["det"].keys())
    fams = {f: [k for k in keys if k.startswith(f + "|")] for f in ("thr", "fix", "ada")}
    glr = [k for k in keys if len(k) > 1 and k[0] == "g" and k[1].isdigit()]
    if glr:
        fams["glr"] = glr
    res = {}
    for turb in sorted({r["turb"] for r in data}):
        for limit in (5, 2):
            row = {}
            for f, ks in fams.items():
                best, b = select(data, ks, turb, limit)
                row[f] = dict(key=best, **(b or {}))
            if row["ada"].get("key") and row["thr"].get("key"):
                row["ada-thr"] = paired_pd_diff(data, turb, row["ada"]["key"], row["thr"]["key"])
            if row.get("glr", {}).get("key") and row["ada"].get("key"):
                row["glr-ada"] = paired_pd_diff(data, turb, row["glr"]["key"], row["ada"]["key"])
                row["glr-thr"] = paired_pd_diff(data, turb, row["glr"]["key"], row["thr"]["key"])
            res[f"{turb}|{limit}"] = row
    return res


# ---------------------------------------------------------------- adaptive-constant sensitivity
def variants():
    data = load("rev2_variants")
    if data is None:
        return None
    keys = list(data[0]["det"].keys())
    names = sorted({k.split("|")[0] for k in keys})
    res = {}
    for turb in (0.5, 0.9):
        for nme in names:
            best, b = select(data, [k for k in keys if k.startswith(nme + "|")], turb, 5)
            res[f"{turb}|{nme}"] = dict(key=best, **(b or {}))
    return res


# ---------------------------------------------------------------- false-alarm interval
def arl():
    data = load("rev2_arl")
    if data is None:
        return None
    rows = []
    for r in data:
        f = lambda d: (d["hours"] * 3600 / d["alarms"]) if d["alarms"] else float("inf")
        rows.append(dict(turb=r["turb"], h0=r["h0"], ada=r["adaptive"], phi=r["stat"]["phi"], sd=r["stat"]["sd"], real=f(r["real"]), ar1=f(r["ar1"]), iid=f(r["iid"]),
                         alarms=(r["real"]["alarms"], r["ar1"]["alarms"], r["iid"]["alarms"])))
    return rows


# ---------------------------------------------------------------- closed loop
def closed(suf=""):
    res = {}
    for est in ("pf", "ekf"):
        for cond in ("calm", "tailwind", "adverse"):
            data = load(f"rev2_closed_{est}_{cond}{suf}")
            if data is None:
                continue
            extra = load(f"rev2_closed_{est}_{cond}_only_kf5s{suf}")
            if extra:
                data = data + [dict(x, name="C5S scheduled Kalman") for x in extra]
            extra = load(f"rev2_closed_{est}_{cond}_only_ki{suf}")
            if extra:
                data = data + [dict(x, name="CI IMU-aided speed-rate Kalman") for x in extra]
            lagd = load(f"rev2_lag_{est}_{cond}{suf}")
            if lagd:
                data = data + lagd
            g = defaultdict(dict)
            for x in data:
                g[x["name"]][x["seed"]] = x
            ref = g.get("C kalman")
            d = {}
            for name, v in g.items():
                fl = [v[s] for s in sorted(v)]
                row = dict(speed=mci([x["speed"] for x in fl]), rej=mci([x["rejected"] for x in fl]), ctr=mci([x["ctr"] for x in fl]), wh=mci([x["wh"] for x in fl]),
                           used=mci([x["used"] for x in fl]), arrived=sum(x["arrived"] for x in fl))
                for key in ("speed", "ctr", "rej"):
                    if name != "C kalman" and ref:
                        dv = np.array([v[s][{"rej": "rejected"}.get(key, key)] - ref[s][{"rej": "rejected"}.get(key, key)] for s in v if v[s][{"rej": "rejected"}.get(key, key)] is not None and ref[s][{"rej": "rejected"}.get(key, key)] is not None])
                        row[f"d_{key}"] = (float(dv.mean()), float(tq(len(dv)) * dv.std(ddof=1) / np.sqrt(len(dv))))
                d[name] = row
            res[f"{est}|{cond}"] = d
    return res


def latency_fit(c):
    """ctr^2 = s0^2 + (k V tau)^2 with tau the Butterworth group delay sqrt(2)/(2 pi fc) and V = 10.5 m/s (circling speed)."""
    out = {}
    for key, d in c.items():
        fcs = [0.1, 0.2, 0.4, 0.8, 1.5]
        y = np.array([d[f"B fc={f}"]["ctr"][0] ** 2 for f in fcs if f"B fc={f}" in d])
        if len(y) < 5:
            continue
        tau = np.sqrt(2) / (2 * np.pi * np.array(fcs))
        X = np.column_stack([np.ones(5), (10.5 * tau) ** 2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss = float(((y - pred) ** 2).sum()); st = float(((y - y.mean()) ** 2).sum())
        # fixed prediction k = 1: only s0 free
        s0sq = float(np.mean(y - (10.5 * tau) ** 2))
        out[key] = dict(s0=float(np.sqrt(max(coef[0], 0))), k=float(np.sqrt(max(coef[1], 0))), r2=1 - ss / st if st > 0 else None,
                        s0_k1=float(np.sqrt(max(s0sq, 0))), pred_k1=[float(np.sqrt(max(s0sq + (10.5 * t) ** 2, 0))) for t in tau], obs=[float(np.sqrt(v)) for v in y])
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def closed_tests(suf=""):
    """Paired t-tests of every configuration against the default Kalman pipeline (same seeds), Holm-adjusted within each estimator and condition."""
    res = {}
    for est in ("pf", "ekf"):
        for cond in ("calm", "tailwind", "adverse"):
            data = load(f"rev2_closed_{est}_{cond}{suf}")
            if data is None:
                continue
            extra = load(f"rev2_closed_{est}_{cond}_only_kf5s{suf}")
            if extra:
                data = data + [dict(x, name="C5S scheduled Kalman") for x in extra]
            extra = load(f"rev2_closed_{est}_{cond}_only_ki{suf}")
            if extra:
                data = data + [dict(x, name="CI IMU-aided speed-rate Kalman") for x in extra]
            g = defaultdict(dict)
            for x in data:
                g[x["name"]][x["seed"]] = x
            ref = g.get("C kalman")
            if not ref:
                continue
            out = {}
            for key, field in (("speed", "speed"), ("ctr", "ctr"), ("rej", "rejected")):
                names, ps, ds = [], [], []
                for name, v in g.items():
                    if name == "C kalman":
                        continue
                    s = [k for k in v if v[k][field] is not None and ref[k][field] is not None]
                    a = np.array([v[k][field] for k in s]); b = np.array([ref[k][field] for k in s])
                    t, p = stats.ttest_rel(a, b)
                    names.append(name); ps.append(p); ds.append(float((a - b).mean()))
                adj = holm(ps)
                out[key] = [(n, d, float(pa)) for n, d, pa in zip(names, ds, adj) if pa < 0.05]
                out[key + "_n"] = len(names)
            res[f"{est}|{cond}"] = out
    return res


# ---------------------------------------------------------------- offline baselines (asymmetric SG, lead-compensated Butterworth)
def baselines():
    sig = load("rev_signals")
    if sig is None:
        return None
    fs = 20.0
    rows = []
    bB, aB = signal.butter(2, 0.4 / (fs / 2))

    def score(fn, use_eh=False):
        per = []; best = {}
        for r in sig:
            w = np.array(r["w"]); x = np.array(r["eh"] if use_eh else r["ed"]); y = fn(x) + np.array(r["sink"])
            per.append(np.sqrt(np.mean((y[300:] - w[300:]) ** 2)))
        return mci(per)

    rows.append(("Butterworth 0.4 Hz", 0.56, score(lambda z: signal.lfilter(bB, aB, z))))
    bb, ab = signal.bessel(2, 0.4 / (fs / 2), norm="mag")
    rows.append(("Bessel 0.4 Hz", 0.54, score(lambda z: signal.lfilter(bb, ab, z))))
    N = int(round(0.443 * fs / 0.4))
    rows.append(("Moving average 0.4 Hz", (N - 1) / 2 / fs, score(lambda z: signal.lfilter(np.ones(N) / N, [1.0], z))))
    al = 1 - np.exp(-2 * np.pi * 0.4 / fs)
    rows.append(("One-pole 0.4 Hz", 1 / (2 * np.pi * 0.4), score(lambda z: signal.lfilter([al], [1, al - 1], z))))
    for tau_c in (0.2, 0.4, 0.56):
        def lead(z, tc=tau_c):
            y = signal.lfilter(bB, aB, z); return y + tc * fs * np.diff(y, prepend=y[0])
        rows.append((f"Butterworth + lead compensation ({tau_c:g} s)", 0.56 - tau_c if tau_c < 0.56 else 0.0, score(lead)))
    for N, p in ((41, 2), (61, 2), (81, 2), (61, 3)):
        for L in (0, 5, 10, 15):
            c = signal.savgol_coeffs(N, p, deriv=1, delta=1 / fs, pos=N - 1 - L)
            rows.append((f"Savitzky-Golay N={N}, p={p}, look-ahead {L / fs:.2f} s", L / fs, score(lambda z, c=c: signal.lfilter(c, [1.0], z), use_eh=True)))
    return rows


SECTIONS = {"pipelines": lambda: {s: pipelines(s) for s in ("", "_real", "_trace")},
            "detect": lambda: {f"{s}|{d}|{suf}": detect(s, d, suf) for s, d in (("kf", 0), ("kf", 2), ("dsp", 2), ("kfl", 2), ("en", 2)) for suf in ("",) if load(f"rev2_detect_{s}_{d}{suf}")},
            "variants": variants, "arl": arl,
            "closed": lambda: {s: closed(s) for s in ("", "_real", "_trace")},
            "baselines": baselines}


def show(name, v):
    print("==", name)
    if name == "pipelines":
        for suf, res in v.items():
            if not res:
                continue
            print(" suffix", repr(suf))
            for cond, d in res.items():
                print("  ", cond, "n", {k: round(x) for k, x in d["_n"].items()})
                for p, x in d.items():
                    if p.startswith("_"):
                        continue
                    print("     %-4s steady %.3f+-%.3f  man %.3f+-%.3f  banked %.3f+-%.3f  bias %+.3f+-%.3f" % (p, *x["steady"], *x["man"], *x["banked"], *x["bias"]))
                if cond in ("nominal", "drift", "tilt1", "polarP"):
                    print("     diffs", {k: tuple(round(y, 3) for y in w) for k, w in d["_diff"].items() if k.endswith("|ma") or k.endswith("|st") or k.endswith("|bn")})
    elif name == "detect":
        for key, res in v.items():
            print(" ", key)
            for k, row in res.items():
                print("    ", k, {f: (x.get("key"), [round(y, 3) for y in x["pd"]] if x.get("pd") else None, [round(y, 2) for y in x["fa"]] if x.get("fa") else None) if isinstance(x, dict) else x for f, x in row.items()})
    elif name == "variants":
        for k, x in v.items():
            print("  ", k, x.get("key"), [round(y, 3) for y in x["pd"]] if x.get("pd") else None, [round(y, 2) for y in x["fa"]] if x.get("fa") else None)
    elif name == "arl":
        for x in v:
            print("  turb %.1f h0 %.1f %s  phi %.3f sd %.2f  ARL(s) real %.0f ar1 %.0f iid %.0f  alarms %s" % (x["turb"], x["h0"], "ada" if x["ada"] else "fix", x["phi"], x["sd"], x["real"], x["ar1"], x["iid"], x["alarms"]))
    elif name == "closed":
        for suf, res in v.items():
            for key, d in res.items():
                print(" ", repr(suf), key)
                for nme, x in d.items():
                    print("     %-38s speed %.2f+-%.2f  rej %.2f+-%.2f  ctr %.1f+-%.1f  wh %.2f  used %.2f  arr %d" % (nme, *x["speed"], *x["rej"], *x["ctr"], x["wh"][0], x["used"][0], x["arrived"]))
                if suf == "":
                    fit = latency_fit({key: d})
                    print("     latency fit", fit)
    elif name == "baselines":
        for nm, dl, (m, c) in v:
            print("   %-52s delay %.2f  RMSE %.3f+-%.3f" % (nm, dl, m, c))


if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS)
    for nme in want:
        res = SECTIONS[nme]()
        OUT[nme] = res
        show(nme, res)
    (R / "rev2_summary.json").write_text(json.dumps(OUT, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
