// Replay of recorded autopilot sensor streams (ALFA dataset, CC0) through the pipelines of the simulator.
//   node tests/real_replay.mjs          (run from E:/AeroSoar/gui; needs data/alfa/streams/segments.json from paper/export_alfa_streams.py)
// The pipelines see only what a flight computer would: barometer, pitot, IMU (rotated with the logged attitude), attitude and GNSS altitude.
// There is no true vertical air velocity in the data, so the output is used by paper/analyze_real_replay.py against an independent GNSS-based reference.
import fs from 'node:fs';
import { RATES, NOISE } from '../js/sim/sensors.js';
import { VarioChain } from '../js/sim/estimation.js';
import { dragSink } from '../js/sim/physics.js';

RATES.baro = 10; RATES.pitot = 10; RATES.imu = 50;
NOISE.baro = 0.25; NOISE.pitot = 0.2; NOISE.accel = 0.26;
const G = 9.80665, D2R = Math.PI / 180;
const segs = JSON.parse(fs.readFileSync('../data/alfa/streams/segments.json'));

const interp = (t, ts, vs) => {
  let lo = 0, hi = ts.length - 1;
  if (t <= ts[0]) return vs[0];
  if (t >= ts[hi]) return vs[hi];
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid; else hi = mid; }
  const f = (t - ts[lo]) / (ts[hi] - ts[lo]);
  return vs[lo] + f * (vs[hi] - vs[lo]);
};

const out = [];
for (const s of segs) {
  const ev = [];
  for (const key of ['imu', 'att', 'arsp', 'baro', 'gps']) for (let i = 0; i < s[key].t.length; i++) ev.push([s[key].t[i], key, i]);
  ev.sort((a, b) => a[0] - b[0]);
  // GNSS altitude is above sea level, the barometer relative to the start: remove the mean offset of the first 20 s
  let off = 0, n = 0;
  for (let i = 0; i < s.gps.t.length; i++) if (s.gps.t[i] < 20 && s.gps.status[i] >= 3) { off += s.gps.alt[i] - interp(s.gps.t[i], s.baro.t, s.baro.alt); n++; }
  off = n ? off / n : 0;
  const chain = new VarioChain();
  chain.reset({ h: s.baro.alt[0], V: s.arsp.v[0], x: 0, y: 0, chi: 0 });
  chain.configure({});
  let roll = 0, pitch = 0;
  const rec = { name: s.name, t: [], A: [], B: [], C: [], C5: [], CL: [], C5L: [], CS: [], D: [], E: [], D2: [], sB: [], sC: [], phi: [] };
  for (const [t, key, i] of ev) {
    if (key === 'att') {
      roll = s.att.roll[i] * D2R; pitch = s.att.pitch[i] * D2R;
      chain.onAhrs({ phi: roll, chi: s.att.yaw[i] * D2R });
    } else if (key === 'imu') {
      const fx = s.imu.ax[i], fy = s.imu.ay[i], fz = s.imu.az[i];
      const fD = -Math.sin(pitch) * fx + Math.sin(roll) * Math.cos(pitch) * fy + Math.cos(roll) * Math.cos(pitch) * fz; // specific force, down component
      chain.onImu({ az: -(fD + G), ax: fx }, 1 / RATES.imu);
    } else if (key === 'arsp') chain.onPitot({ V: s.arsp.v[i] });
    else if (key === 'gps') {
      if (s.gps.status[i] >= 3) chain.onGnss({ h: s.gps.alt[i] - off, x: 0, y: 0, vx: 0, vy: 0, tAcq: t }, t);
    } else {
      chain.onBaro({ h: s.baro.alt[i] }, false, 0);
      if (t < s.t_start) continue;
      const phi = chain.nav.phi;
      rec.t.push(+t.toFixed(3));
      const r = (x) => +x.toFixed(4);
      rec.A.push(r(chain.wRaw)); rec.B.push(r(chain.wDsp)); rec.C.push(r(chain.wKf)); rec.C5.push(r(chain.wKf5)); rec.CL.push(r(chain.wKfL)); rec.C5L.push(r(chain.wKf5L));
      rec.CS.push(r(chain.wKf5S)); rec.D.push(r(chain.wEn)); rec.E.push(r(chain.wComp)); rec.D2.push(r(chain.wEnA));
      rec.sB.push(r(dragSink(Math.max(6, chain.Vlp), phi))); rec.sC.push(r(dragSink(Math.max(6, chain.akf.x[0]), phi))); rec.phi.push(r(phi));
    }
  }
  out.push(rec);
}
fs.writeFileSync('../data/alfa/streams/replay_out.json', JSON.stringify(out));
console.log('replayed', out.length, 'segments,', out.reduce((a, r) => a + r.t.length, 0), 'baro samples');
