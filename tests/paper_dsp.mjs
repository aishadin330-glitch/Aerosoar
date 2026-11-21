// DSP experiments for the paper.   node tests/paper_dsp.mjs
//   filters   : filter type x cutoff, error and measured delay of the vertical-air-velocity estimate
//   detectors : four detectors on the same signal, false alarms per hour and detection latency
//   series    : one 300 s record of the true and estimated vertical air velocity (for spectra)
import { Sim, defaultConfig, LIMITS } from '../js/sim/sim.js';
import { Cusum } from '../js/sim/estimation.js';
import fs from 'node:fs';

const SEEDS = Array.from({ length: 30 }, (_, i) => 100 + i);
const out = { filters: [], detectors: [], ablation: [], series: null };
const FS = 20;

/** lag [s] in 0..3 that minimises the mean-square difference between y and truth */
function bestLag(y, w) {
  let best = 0, bv = 1e9;
  for (let l = 0; l <= 3 * FS; l++) {
    let s = 0, n = 0;
    for (let i = 10 * FS; i + l < y.length; i++) { s += (y[i + l] - w[i]) ** 2; n++; }
    if (s / n < bv) { bv = s / n; best = l; }
  }
  return best / FS;
}

// ---- ablation of the total-energy term and of the Kalman fusion: full missions with guidance on, so speed changes occur.
// Samples are split by the true speed rate: steady (|dV/dt| < 0.15 m/s^2) and manoeuvring.
for (const cond of ['nominal', 'baro drift'])
  for (const seed of SEEDS) {
    const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'pf';
    if (cond === 'baro drift') cfg.faults.baro = true;
    const sim = new Sim(cfg);
    const z = () => ({ raw: 0, noTE: 0, te: 0, kf: 0, n: 0 });
    const acc = { steady: z(), manoeuvre: z() };
    let last = 0, vPrev = sim.glider.V;
    for (let i = 0; i < 300 * 100 && !sim.done; i++) {
      sim.step();
      if (sim.vt !== last) {
        const vd = (sim.glider.V - vPrev) / 0.05; vPrev = sim.glider.V; last = sim.vt;
        if (sim.t < 10) continue;
        const a = Math.abs(vd) < 0.15 ? acc.steady : acc.manoeuvre, w = sim.air.w, c = sim.chain;
        a.raw += (c.wRaw - w) ** 2; a.noTE += (c.wNoTE - w) ** 2; a.te += (c.wDsp - w) ** 2; a.kf += (c.wKf - w) ** 2; a.n++;
      }
    }
    const r = (a) => ({ raw: Math.sqrt(a.raw / a.n), noTE: Math.sqrt(a.noTE / a.n), te: Math.sqrt(a.te / a.n), kf: Math.sqrt(a.kf / a.n), n: a.n });
    out.ablation.push({ cond, seed, steady: r(acc.steady), manoeuvre: r(acc.manoeuvre) });
  }

// ---- E1: filters.  Cruise only, so every filter setting sees the identical flight.
LIMITS.ceiling = 0; // topOut is always true: no thermal climbs
LIMITS.reentry = -1;
for (const type of ['butter', 'movavg', 'ema'])
  for (const fc of [0.1, 0.2, 0.4, 0.8, 1.5])
    for (const seed of SEEDS) {
      const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.filter = type; cfg.fc = fc; cfg.estimator = 'oracle';
      const sim = new Sim(cfg);
      const w = [], y = [], ynote = [];
      let last = 0;
      for (let i = 0; i < 240 * 100; i++) {
        sim.step();
        if (sim.vt !== last) { last = sim.vt; w.push(sim.air.w); y.push(sim.chain.wDsp); ynote.push(sim.chain.wNoTE); }
      }
      let e = 0, n = 0;
      for (let i = 10 * FS; i < y.length; i++) { e += (y[i] - w[i]) ** 2; n++; }
      out.filters.push({ type, fc, seed, rmse: Math.sqrt(e / n), delay: bestLag(y, w), delayTheory: sim.chain.delay });
    }

// ---- E3: detectors, straight flight through the thermals, 300 s per flight
function makeDetectors() {
  const thr = { n: 0, lock: 0, step(w, dt) { if (this.lock > 0) { this.lock -= dt; this.n = 0; return false; } this.n = w > 1.0 ? this.n + dt : 0; if (this.n >= 2) { this.lock = 8; this.n = 0; return true; } return false; } };
  const mk = (adaptive) => {
    const c = new Cusum();
    return { c, lock: 0, step(w, dt) {
      if (!adaptive) c.gust = 0;
      const hit = c.step(w, dt, this.lock <= 0);
      if (this.lock > 0) this.lock -= dt;
      if (hit) { this.lock = 8; c.reset(); return true; }
      return false;
    } };
  };
  return { 'Threshold, 1 m/s for 2 s': thr, 'CUSUM, fixed': mk(false), 'CUSUM, gust-adaptive': mk(true), 'CUSUM on unfiltered w': mk(false) };
}
for (const turb of [0.2, 0.5, 0.9])
  for (const seed of SEEDS) {
    const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'oracle';
    const sim = new Sim(cfg);
    sim.atm.turb = turb;
    // the scenario fixes turbulence at reset; override the process directly
    const dets = makeDetectors();
    const st = {};
    for (const k in dets) st[k] = { fa: 0, hit: 0, lat: [], armedEnc: false };
    let truthIn = false, tEnter = 0, tLeave = -99, enc = 0, last = 0;
    const encDetected = {};
    for (let i = 0; i < 300 * 100; i++) {
      sim.step();
      if (sim.vt === last) continue;
      last = sim.vt;
      const inLift = sim.air.wT > 0.6;
      if (inLift && !truthIn) { tEnter = sim.t; enc++; for (const k in dets) encDetected[k + enc] = false; }
      if (!inLift && truthIn) tLeave = sim.t;
      truthIn = inLift;
      for (const k in dets) {
        const w = k === 'CUSUM on unfiltered w' ? sim.chain.wRaw : sim.w;
        if (!dets[k].step(w, 0.05)) continue;
        const near = truthIn || sim.t - tLeave < 5;
        if (near && !encDetected[k + enc] && enc > 0) { encDetected[k + enc] = true; st[k].hit++; st[k].lat.push(Math.max(0, sim.t - tEnter)); } else if (!near) st[k].fa++;
      }
    }
    for (const k in dets) out.detectors.push({ det: k, turb, seed, enc, fa: st[k].fa, hit: st[k].hit, lat: st[k].lat, t: sim.t });
  }

// ---- one record for spectra
{
  const cfg = defaultConfig('tailwind'); cfg.seed = 100; cfg.estimator = 'oracle';
  const sim = new Sim(cfg);
  const s = { w: [], A: [], B: [], C: [] };
  let last = 0;
  for (let i = 0; i < 300 * 100; i++) {
    sim.step();
    if (sim.vt !== last) { last = sim.vt; s.w.push(sim.air.w); s.A.push(sim.chain.wRaw); s.B.push(sim.chain.wDsp); s.C.push(sim.chain.wKf); }
  }
  out.series = s;
}
fs.writeFileSync('../paper/results/dsp.json', JSON.stringify(out));
console.log('done', out.filters.length, out.detectors.length, out.ablation.length);
