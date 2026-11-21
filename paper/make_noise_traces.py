"""Extract recorded sensor-noise traces from the ALFA flights (CC0) for replay in the simulator.
usage (project root): uv run --with pymavlink --with scipy --with numpy python -I paper/make_noise_traces.py
The trace of each sensor is the residual of an airborne record after removing a smooth trend (Savitzky-Golay, about 3 s for the
barometer and airspeed, 1 s for the accelerometer), so it holds the recorded noise above roughly 0.2 Hz, with its true colour and tails.
Writes paper/results/alfa_noise_traces.json and prints the statistics used in the paper."""
import glob, json, pathlib, collections
import numpy as np
from scipy import signal, stats
from pymavlink import DFReader

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted(glob.glob(str(ROOT / "data/alfa/dataflash/dataflash/*/*.bin")))
WANT = {"BARO": ("TimeUS", "Alt"), "ARSP": ("TimeUS", "Airspeed"), "IMU": ("TimeUS", "AccZ")}


def read(path):
    log = DFReader.DFReader_binary(path)
    d = collections.defaultdict(list)
    while True:
        m = log.recv_msg()
        if m is None:
            break
        t = m.get_type()
        if t in WANT:
            try:
                d[t].append([getattr(m, k) for k in WANT[t]])
            except AttributeError:
                pass
    return {k: np.array(v, float) for k, v in d.items()}


def runs(mask, fs, min_s=60):
    """Contiguous True runs of at least min_s seconds."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) / fs >= min_s:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def residual(x, fs, win_s, order=3):
    n = int(win_s * fs) | 1
    return x - signal.savgol_filter(x, n, order)


traces = {"baro": [], "pitot": [], "accel": []}
for f in FILES:
    d = read(f)
    if not all(k in d and len(d[k]) > 500 for k in WANT):
        continue
    tb, alt = d["BARO"][:, 0] / 1e6, d["BARO"][:, 1]
    ta, asp = d["ARSP"][:, 0] / 1e6, d["ARSP"][:, 1]
    ti, az = d["IMU"][:, 0] / 1e6, d["IMU"][:, 1]
    fs = lambda t: 1.0 / np.median(np.diff(t))
    for key, t, x, win in (("baro", tb, alt, 3.0), ("pitot", ta, asp, 3.0), ("accel", ti, az, 1.0)):
        fly = np.interp(t, ta, asp) > 10.0
        for a, b in runs(fly, fs(t)):
            traces[key].extend(residual(x[a:b], fs(t), win).tolist())

out = {k: [round(v, 4) for v in vals] for k, vals in traces.items()}
(ROOT / "paper" / "results" / "alfa_noise_traces.json").write_text(json.dumps(out))
for k, v in traces.items():
    v = np.array(v)
    ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    print(f"{k}: n={len(v)} std={v.std():.3f} lag1_autocorr={ac1:.2f} excess_kurtosis={stats.kurtosis(v):.2f} p99.9/std={np.percentile(np.abs(v), 99.9) / v.std():.1f}")
