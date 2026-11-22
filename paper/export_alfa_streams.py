"""Export airborne segments of the ALFA autopilot logs (CC0) as plain sensor streams for replay through the JavaScript pipelines.
usage (project root): uv run --with pymavlink --with numpy python -I paper/export_alfa_streams.py
Writes data/alfa/streams/segments.json (not part of the repository: regenerate it from the downloaded dataflash.zip).
A segment is a contiguous airborne run (pitot speed above 10 m/s for at least 120 s) with 25 s of lead-in for filter start-up."""
import glob, json, pathlib, collections
import numpy as np
from pymavlink import DFReader

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted(glob.glob(str(ROOT / "data/alfa/dataflash/dataflash/*/*.bin")))
WANT = {"BARO": ("TimeUS", "Alt"), "ARSP": ("TimeUS", "Airspeed"), "IMU": ("TimeUS", "AccX", "AccY", "AccZ"), "ATT": ("TimeUS", "Roll", "Pitch", "Yaw"),
        "GPS": ("TimeUS", "Alt", "VZ", "Spd", "Status")}


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


segments = []
for f in FILES:
    d = read(f)
    if not all(k in d and len(d[k]) > 500 for k in WANT):
        continue
    tb = d["BARO"][:, 0] / 1e6
    ta, asp = d["ARSP"][:, 0] / 1e6, d["ARSP"][:, 1]
    fly = asp > 10.0
    i = 0
    n = len(fly)
    while i < n:
        if fly[i]:
            j = i
            while j < n and fly[j]:
                j += 1
            if ta[j - 1] - ta[i] >= 120:
                t0, t1 = ta[i] - 25.0, ta[j - 1]
                seg = {"name": pathlib.Path(f).stem, "t_start": 25.0, "dur": float(t1 - t0)}
                for key, cols in (("baro", ("t", "alt")), ("arsp", ("t", "v")), ("imu", ("t", "ax", "ay", "az")), ("att", ("t", "roll", "pitch", "yaw")), ("gps", ("t", "alt", "vz", "spd", "status"))):
                    arr = d[{"baro": "BARO", "arsp": "ARSP", "imu": "IMU", "att": "ATT", "gps": "GPS"}[key]]
                    tt = arr[:, 0] / 1e6
                    m = (tt >= t0) & (tt <= t1)
                    seg[key] = {c: [round(float(v), 4) for v in (tt[m] - t0 if c == "t" else arr[m][:, k])] for k, c in enumerate(cols)}
                segments.append(seg)
            i = j
        else:
            i += 1

out = ROOT / "data" / "alfa" / "streams"
out.mkdir(parents=True, exist_ok=True)
(out / "segments.json").write_text(json.dumps(segments))
print(len(segments), "segments,", round(sum(s["dur"] - 25 for s in segments) / 60, 1), "min scored;", [round(s["dur"]) for s in segments])
