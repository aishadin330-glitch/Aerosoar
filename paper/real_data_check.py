"""Compare the assumed sensor model with recorded ALFA flights (CC0, Keipour et al., CMU).

usage (project root):  uv run --with pymavlink --with scipy --with numpy python -I paper/real_data_check.py
Reads data/alfa/dataflash/dataflash/*/*.bin (ArduPlane DataFlash logs); writes paper/results/real_alfa.json.
Only airborne samples (airspeed above 10 m/s) are used. No true vertical air velocity exists in recorded data, so the
filtered climb rate is compared with the GNSS vertical velocity, which is a reference, not the truth.
"""
import glob, json, pathlib, collections
import numpy as np
from scipy import signal
from pymavlink import DFReader

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted(glob.glob(str(ROOT / "data/alfa/dataflash/dataflash/*/*.bin")))


def read(path):
    log = DFReader.DFReader_binary(path)
    d = collections.defaultdict(list)
    want = {"BARO": ("TimeUS", "Alt"), "ARSP": ("TimeUS", "Airspeed"), "IMU": ("TimeUS", "AccZ"), "GPS": ("TimeUS", "VZ", "Spd")}
    while True:
        m = log.recv_msg()
        if m is None:
            break
        t = m.get_type()
        if t in want and (t != "GPS" or getattr(m, "I", 0) == 0 or True):
            try:
                d[t].append([getattr(m, k) for k in want[t]])
            except AttributeError:
                pass
    return {k: np.array(v, float) for k, v in d.items()}


def mad_sigma(x, k):
    """White-noise standard deviation from the k-th difference (variance factor 6 for k=2)."""
    return 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(6.0 if k == 2 else 2.0)


def second_diff_sigma(x):
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return mad_sigma(d2, 2)


rows = []
for f in FILES:
    d = read(f)
    if not all(k in d and len(d[k]) > 500 for k in ("BARO", "ARSP", "IMU", "GPS")):
        continue
    tb, alt = d["BARO"][:, 0] / 1e6, d["BARO"][:, 1]
    ta, asp = d["ARSP"][:, 0] / 1e6, d["ARSP"][:, 1]
    ti, az = d["IMU"][:, 0] / 1e6, d["IMU"][:, 1]
    tg, vz, spd = d["GPS"][:, 0] / 1e6, d["GPS"][:, 1], d["GPS"][:, 2]
    fly_b = np.interp(tb, ta, asp) > 10.0
    fly_a = asp > 10.0
    fly_i = np.interp(ti, ta, asp) > 10.0
    if fly_b.sum() < 600:
        continue
    fsb = 1.0 / np.median(np.diff(tb))
    fsa = 1.0 / np.median(np.diff(ta))
    fsi = 1.0 / np.median(np.diff(ti))
    fsg = 1.0 / np.median(np.diff(tg))
    # white-noise level from the second difference of airborne segments
    sb = second_diff_sigma(alt[fly_b])
    sa = second_diff_sigma(asp[fly_a])
    si = second_diff_sigma(az[fly_i])
    # spectral noise floor of the barometric altitude (mean PSD between 0.6 and 1.0 of Nyquist)
    seg = alt[fly_b]
    fr, P = signal.welch(signal.detrend(seg), fs=fsb, nperseg=min(256, len(seg) // 4))
    hi = (fr > 0.6 * fsb / 2)
    floor = float(np.mean(P[hi]))
    sb_psd = float(np.sqrt(floor * fsb / 2))
    # climb rate: plain difference (A) and 2nd-order Butterworth at 0.4 Hz (B-like), against GNSS vertical velocity
    vA = np.gradient(alt, tb)
    b, a = signal.butter(2, 0.4 / (fsb / 2))
    vB = signal.lfilter(b, a, vA)
    ref = np.interp(tb, tg, vz)
    sgn = 1.0 if np.corrcoef(vB[fly_b], ref[fly_b])[0, 1] > 0 else -1.0
    ref = sgn * ref
    ok = fly_b.copy()
    ok[: int(30 * fsb)] = False
    ok &= np.isfinite(vA) & np.isfinite(ref)
    best = None
    for sh in range(0, int(5 * fsb)):  # allow the filter delay: shift the reference later
        r = np.sqrt(np.mean((vB[ok][sh:] - ref[ok][: len(ref[ok]) - sh]) ** 2))
        best = r if best is None or r < best else best
    rows.append(dict(file=pathlib.Path(f).name, airborne_s=float(fly_b.sum() / fsb), fs_baro=fsb, fs_arsp=fsa, fs_imu=fsi, fs_gps=fsg,
                     sigma_baro=sb, sigma_baro_psd=sb_psd, sigma_airspeed=sa, sigma_accz=si,
                     rms_A_vs_gps=float(np.sqrt(np.mean((vA[ok] - ref[ok]) ** 2))), rms_B_vs_gps=float(np.sqrt(np.mean((vB[ok] - ref[ok]) ** 2))),
                     rms_B_vs_gps_aligned=float(best), gps_vz_std=float(np.std(ref[ok])), diff_noise_pred=float(np.sqrt(2) * sb * fsb)))
    print(rows[-1]["file"], {k: round(v, 3) for k, v in rows[-1].items() if k != "file"}, flush=True)

out = ROOT / "paper" / "results" / "real_alfa.json"
out.write_text(json.dumps(rows))
med = lambda k: float(np.median([r[k] for r in rows]))
print("MEDIAN over", len(rows), "logs:", {k: round(med(k), 3) for k in rows[0] if k != "file"})
