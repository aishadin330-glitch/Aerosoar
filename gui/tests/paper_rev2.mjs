// Revision-2 experiments.   node tests/paper_rev2.mjs <mode> [args]       (run from E:/AeroSoar/gui)
//   pipelines             all pipelines on the same guided flights, several conditions; strata: level, banked, speed change
//   detect <source> <dbl> detector operating curves (finer grids), tuning seeds 500-529, test seeds 1000-1119; dbl = speed doublet (m/s)
//   variants              sensitivity of the adaptive-CUSUM constants, one at a time
//   arl                   false-alarm interval in thermal-free air: recorded flights of the simulator vs AR(1) and i.i.d. references
//   closed <est> <cond>   closed-loop 10 km missions: cutoff sweep and the other pipelines
// Environment: REAL=1 sets the sensor rates and noise levels measured in the ALFA flights;
//              TRACE=1 (implies REAL) additionally replays recorded barometer and accelerometer noise.
// Output files get the suffix _real or _trace and are never written over the files of the earlier experiments.
import { Sim, defaultConfig, LIMITS } from '../js/sim/sim.js';
import { Cusum } from '../js/sim/estimation.js';
import { RATES, NOISE } from '../js/sim/sensors.js';
import { RNG } from '../js/sim/util.js';
import fs from 'node:fs';

const mode = process.argv[2];
const arg1 = process.argv[3] || '';
const arg2 = process.argv[4] || '';
const TRACE = process.env.TRACE === '1';
const REAL = process.env.REAL === '1' || TRACE;
if (REAL) { RATES.baro = 10; RATES.pitot = 10; RATES.imu = 50; NOISE.baro = 0.25; NOISE.pitot = 0.2; NOISE.accel = 0.26; }
const SUF = TRACE ? '_trace' : REAL ? '_real' : '';
let TR = null;
if (TRACE) { const t = JSON.parse(fs.readFileSync('../paper/results/alfa_noise_traces.json')); TR = { baro: t.baro, accel: t.accel }; }
const DTB = 1 / RATES.baro;
const OUT = '../paper/results/';
const SEEDS = Array.from({ length: 30 }, (_, i) => 100 + i);
const CEIL = { ceiling: LIMITS.ceiling, reentry: LIMITS.reentry };
const cruiseOnly = (on) => { LIMITS.ceiling = on ? 0 : CEIL.ceiling; LIMITS.reentry = on ? -1 : CEIL.reentry; };
const save = (name, data) => fs.writeFileSync(OUT + name + SUF + '.json', JSON.stringify(data));
const mk = (cond, seed) => { const cfg = defaultConfig(cond); cfg.seed = seed; if (TR) cfg.traces = TR; return cfg; };
const out = [];

// ------------------------------------------------------------------ detectors
function threshold(level) {
  return { n: 0, lock: 0, step(w, dt) { if (this.lock > 0) { this.lock -= dt; this.n = 0; return false; } this.n = w > level ? this.n + dt : 0; if (this.n >= 2) { this.lock = 8; this.n = 0; return true; } return false; } };
}
function cusum(adaptive, h0, c = {}) {
  const d = new Cusum(0.5, h0);
  Object.assign(d, c);
  return { lock: 0, step(w, dt) {
    if (!adaptive) d.gust = 0;
    const hit = d.step(w, dt, this.lock <= 0);
    if (this.lock > 0) this.lock -= dt;
    if (hit) { this.lock = 8; d.reset(); return true; }
    return false;
  } };
}
/** Matched-filter (GLR-type) detector: zero-mean rising half-Gaussian template of width sigT, peak at the newest sample, window W seconds;
 *  the correlation is normalised by its own running standard deviation (60 s), so one threshold is used in every turbulence level. */
function matched(sigT, thr, W = 12) {
  const N = Math.round(W / DTB);
  const s = Array.from({ length: N }, (_, j) => Math.exp(-(((N - 1 - j) * DTB) ** 2) / (2 * sigT * sigT)));
  const mu = s.reduce((a, b) => a + b, 0) / N;
  const z = s.map((v) => v - mu);
  const nrm = Math.sqrt(z.reduce((a, v) => a + v * v, 0));
  const buf = new Float64Array(N);
  let head = 0, filled = 0, v = 0.09, lock = 0;
  return { step(w, dt) {
    buf[head] = w; head = (head + 1) % N; if (filled < N) filled++;
    if (filled < N) return false;
    let c = 0;
    for (let j = 0; j < N; j++) c += z[j] * buf[(head + j) % N];
    c /= nrm;
    v += ((c * c - v) * dt) / 60;
    if (lock > 0) { lock -= dt; return false; }
    if (c / Math.sqrt(v) > thr) { lock = 8; return true; }
    return false;
  } };
}
const THR = [0.6, 0.8, 1, 1.2, 1.5, 1.8, 2, 2.25, 2.5, 2.75, 3];
const HH = [0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8];
const bank = () => {
  const d = {};
  for (const l of THR) d[`thr|${l}`] = threshold(l);
  for (const h of HH) { d[`fix|${h}`] = cusum(false, h); d[`ada|${h}`] = cusum(true, h); }
  if (process.env.MATCHED === '1') for (const sg of [2, 3, 4]) for (const th of [1.5, 2, 2.5, 3, 3.5, 4, 5, 6]) d[`g${sg}|${th}`] = matched(sg, th);
  return d;
};

/** One 300 s straight flight through the thermals; every detector sees the same estimator output sim.w. */
function detectionFlight(seed, turb, extra, dets) {
  const cfg = mk('tailwind', seed); cfg.estimator = 'oracle'; extra(cfg);
  const sim = new Sim(cfg);
  sim.atm.turb = turb;
  const st = {}, hit = {};
  for (const k in dets) st[k] = { hit: 0, fa: 0, lat: [] };
  let truthIn = false, tEnter = 0, tLeave = -99, enc = 0, last = 0;
  for (let i = 0; i < 300 * 100; i++) {
    sim.step();
    if (sim.vt === last) continue;
    last = sim.vt;
    const inLift = sim.air.wT > 0.6;
    if (inLift && !truthIn) { tEnter = sim.t; enc++; for (const k in dets) hit[k + enc] = false; }
    if (!inLift && truthIn) tLeave = sim.t;
    truthIn = inLift;
    for (const k in dets) {
      if (!dets[k].step(sim.w, DTB)) continue;
      const near = truthIn || sim.t - tLeave < 5;
      if (near && enc > 0 && !hit[k + enc]) { hit[k + enc] = true; st[k].hit++; st[k].lat.push(Math.max(0, sim.t - tEnter)); } else if (!near) st[k].fa++;
    }
  }
  return { seed, turb, enc, t: sim.t, det: st };
}

// ------------------------------------------------------------------ modes
if (mode === 'pipelines') {
  cruiseOnly(false);
  const conds = {
    nominal: () => {},
    drift: (c) => { c.faults.baro = true; },
    polarM: (c) => { c.polarErr = -0.2; },
    polarP: (c) => { c.polarErr = 0.2; },
    tilt1: (c) => { c.stress = { tilt: 1 }; },
    tilt1Q: (c) => { c.stress = { tilt: 1 }; c.tune = { kf: { tiltQ: true } }; },
    tilt1B: (c) => { c.stress = { tilt: 1 }; c.tune = { kf: { tiltQ: 'bias' } }; },
    aoa05: (c) => { c.stress = { aoa: 0.5 }; },
    aoa1: (c) => { c.stress = { aoa: 1 }; },
    gnss: (c) => { c.faults.gnss = true; },
    pitot: (c) => { c.faults.pitot = true; },
    noise2: (c) => { c.noise = 2; },
    noise3: (c) => { c.noise = 3; },
  };
  for (const [name, f] of Object.entries(conds))
    for (const seed of SEEDS) {
      const cfg = mk('tailwind', seed); cfg.estimator = 'oracle'; f(cfg);
      const sim = new Sim(cfg);
      const P = ['A', 'B', 'C', 'C5', 'CL', 'C5L', 'CS', 'C5S', 'D', 'D2', 'E', 'Cx'];
      const S = ['lvl', 'ban', 'man'];
      const se = {}, sb = {}, n = { lvl: 0, ban: 0, man: 0 };
      for (const p of P) { se[p] = { lvl: 0, ban: 0, man: 0 }; sb[p] = 0; }
      let last = 0, vPrev = sim.glider.V, nn = 0;
      for (let i = 0; i < 300 * 100 && !sim.done; i++) {
        sim.step();
        if (sim.vt === last) continue;
        last = sim.vt;
        const vd = (sim.glider.V - vPrev) / DTB; vPrev = sim.glider.V;
        if (sim.t < 15) continue;
        const ch = sim.chain, w = sim.air.w;
        const s = Math.abs(vd) >= 0.15 ? 'man' : Math.abs(sim.glider.phi) > 0.35 ? 'ban' : 'lvl';
        n[s]++; nn++;
        // Cx is a diagnostic only: pipeline C with the true speed rate in place of its airspeed-filter estimate
        const Vk = Math.max(6, ch.akf.x[0]);
        const v = { A: ch.wRaw, B: ch.wDsp, C: ch.wKf, C5: ch.wKf5, CL: ch.wKfL, C5L: ch.wKf5L, CS: ch.wKfS, C5S: ch.wKf5S, D: ch.wEn, D2: ch.wEnA, E: ch.wComp, Cx: ch.wKf - (Vk * ch.akf.x[1]) / 9.80665 + (Vk * vd) / 9.80665 };
        for (const p of P) { const e = v[p] - w; se[p][s] += e * e; sb[p] += e; }
      }
      out.push({ cond: name, seed, n, nn, se, bias: Object.fromEntries(P.map((p) => [p, sb[p] / nn])) });
    }
  save('rev2_pipelines', out);
} else if (mode === 'tune') {
  // tuning of the new filters on separate seeds (500-529): energy-rate jerk density qe and complementary crossover fcc
  cruiseOnly(false);
  const grid = [];
  for (const qe of [0.05, 0.1, 0.15, 0.2, 0.3]) grid.push({ qe, fcc: 0.12 });
  for (const fcc of [0.03, 0.06, 0.09, 0.18, 0.4]) grid.push({ qe: 0.2, fcc });
  for (const kf of grid)
    for (let seed = 500; seed < 530; seed++) {
      const cfg = mk('tailwind', seed); cfg.estimator = 'oracle'; cfg.tune = { kf };
      const sim = new Sim(cfg);
      let last = 0, vPrev = sim.glider.V;
      const a = { D: [0, 0, 0, 0], D2: [0, 0, 0, 0], E: [0, 0, 0, 0] };
      for (let i = 0; i < 300 * 100 && !sim.done; i++) {
        sim.step();
        if (sim.vt === last) continue;
        last = sim.vt;
        const vd = (sim.glider.V - vPrev) / DTB; vPrev = sim.glider.V;
        if (sim.t < 15) continue;
        const ch = sim.chain, w = sim.air.w, man = Math.abs(vd) >= 0.15 ? 2 : 0;
        for (const [p, v] of [['D', ch.wEn], ['D2', ch.wEnA], ['E', ch.wComp]]) { a[p][man] += (v - w) ** 2; a[p][man + 1]++; }
      }
      out.push({ ...kf, seed, a });
    }
  save('rev2_tune', out);
} else if (mode === 'tuneS') {
  // tuning of the scheduled blend on separate seeds (500-529)
  cruiseOnly(false);
  const grid = [];
  for (const sA0 of [0.06, 0.1]) for (const sA1 of [0.3, 0.4, 0.6]) for (const sHold of [0.5, 1]) grid.push({ sA0, sA1, sHold });
  for (const kf of grid)
    for (let seed = 500; seed < 530; seed++) {
      const cfg = mk('tailwind', seed); cfg.estimator = 'oracle'; cfg.tune = { kf };
      const sim = new Sim(cfg);
      let last = 0, vPrev = sim.glider.V;
      const a = [0, 0, 0, 0];
      for (let i = 0; i < 300 * 100 && !sim.done; i++) {
        sim.step();
        if (sim.vt === last) continue;
        last = sim.vt;
        const vd = (sim.glider.V - vPrev) / DTB; vPrev = sim.glider.V;
        if (sim.t < 15) continue;
        const man = Math.abs(vd) >= 0.15 ? 2 : 0;
        a[man] += (sim.chain.wKfS - sim.air.w) ** 2; a[man + 1]++;
      }
      out.push({ ...kf, seed, a });
    }
  save('rev2_tuneS', out);
} else if (mode === 'detect') {
  cruiseOnly(true);
  const src = arg1 || 'kf', dbl = +(arg2 || 0);
  const turbs = dbl ? [0.5] : [0.2, 0.5, 0.9];
  const plan = [['tune', Array.from({ length: 30 }, (_, i) => 500 + i)], ['test', Array.from({ length: 120 }, (_, i) => 1000 + i)]];
  for (const [split, seeds] of plan)
    for (const turb of turbs)
      for (const seed of seeds) out.push({ split, ...detectionFlight(seed, turb, (c) => { c.source = src; if (dbl) c.speedDoublet = dbl; }, bank()) });
  save(`rev2_detect_${src}_${dbl}`, out);
} else if (mode === 'variants') {
  cruiseOnly(true);
  const base = { ak: 0.7, ah: 1.2, ag: 1.25, tb: 6, tg: 25 };
  const variants = { default: {} };
  for (const [k, v] of Object.entries(base)) for (const f of [0.5, 2]) variants[`${k}x${f}`] = { [k]: v * f };
  const mkBank = () => {
    const d = {};
    for (const [name, c] of Object.entries(variants)) for (const h of HH) d[`${name}|${h}`] = cusum(true, h, c);
    return d;
  };
  const plan = [['tune', Array.from({ length: 30 }, (_, i) => 500 + i)], ['test', Array.from({ length: 120 }, (_, i) => 1000 + i)]];
  for (const [split, seeds] of plan)
    for (const turb of [0.5, 0.9])
      for (const seed of seeds) out.push({ split, ...detectionFlight(seed, turb, (c) => { c.source = 'kf'; }, mkBank()) });
  save('rev2_variants', out);
} else if (mode === 'arl') {
  // thermal-free air: every alarm is false. The estimator output is recorded, then the detectors are run offline on it.
  cruiseOnly(true);
  const series = {};
  for (const turb of [0.2, 0.5, 0.9]) {
    series[turb] = [];
    for (let s = 0; s < 40; s++) {
      const cfg = mk('tailwind', 2000 + s); cfg.estimator = 'oracle'; cfg.source = 'kf';
      const sim = new Sim(cfg);
      sim.atm.thermals = [];
      sim.atm.turb = turb;
      const w = []; let last = 0;
      for (let i = 0; i < 700 * 100; i++) { sim.step(); if (sim.vt !== last) { last = sim.vt; if (sim.t > 40) w.push(sim.w); } }
      series[turb].push(w);
    }
  }
  const stat = (arrs) => {
    let n = 0, m = 0, v = 0, c1 = 0;
    for (const a of arrs) for (const x of a) { n++; m += x; }
    m /= n;
    for (const a of arrs) for (let i = 0; i < a.length; i++) { v += (a[i] - m) ** 2; if (i) c1 += (a[i] - m) * (a[i - 1] - m); }
    return { mean: m, sd: Math.sqrt(v / n), phi: c1 / v, n };
  };
  const run = (arrs, make) => { let al = 0, T = 0; for (const a of arrs) { const d = make(); for (const x of a) if (d.step(x, DTB)) al++; T += a.length * DTB; } return { alarms: al, hours: T / 3600 }; };
  const rng = new RNG(4242);
  const syn = (st, phi, len, count) => {
    const arrs = [];
    for (let j = 0; j < count; j++) {
      const a = []; let x = 0;
      for (let i = 0; i < len; i++) { x = phi * x + Math.sqrt(1 - phi * phi) * rng.gauss(1); a.push(st.mean + st.sd * x); }
      arrs.push(a);
    }
    return arrs;
  };
  for (const turb of [0.2, 0.5, 0.9]) {
    const st = stat(series[turb]);
    const len = series[turb][0].length;
    const ar = syn(st, st.phi, len, 40), iid = syn(st, 0, len, 40);
    for (const h of [1, 1.5, 2, 3, 4, 6])
      for (const ad of [false, true]) {
        const mkd = () => cusum(ad, h);
        out.push({ turb, h0: h, adaptive: ad, stat: st, real: run(series[turb], mkd), ar1: run(ar, mkd), iid: run(iid, mkd) });
      }
  }
  save('rev2_arl', out);
} else if (mode === 'artefact') {
  // size of the difference between pipelines, by speed-change stratum, for comparison with the replay of recorded flights
  cruiseOnly(false);
  for (const seed of SEEDS) {
    const cfg = mk('tailwind', seed); cfg.estimator = 'oracle';
    const sim = new Sim(cfg);
    const S = { steady: { cl: 0, bl: 0, n: 0 }, man: { cl: 0, bl: 0, n: 0 } };
    let last = 0, vPrev = sim.glider.V, vd2 = 0, nn = 0;
    for (let i = 0; i < 300 * 100 && !sim.done; i++) {
      sim.step();
      if (sim.vt === last) continue;
      last = sim.vt;
      const vd = (sim.glider.V - vPrev) / DTB; vPrev = sim.glider.V;
      if (sim.t < 15) continue;
      const ch = sim.chain, s = Math.abs(vd) >= 0.15 ? 'man' : 'steady';
      S[s].cl += (ch.wKf - ch.wKfL) ** 2; S[s].bl += (ch.wDsp - ch.wKfL) ** 2; S[s].n++;
      vd2 += vd * vd; nn++;
    }
    out.push({ seed, S, vdRms: Math.sqrt(vd2 / nn) });
  }
  save('rev2_artefact', out);
} else if (mode === 'lag') {
  // closed loop with delay-compensated measurement assignment: posLag = group delay of the Butterworth filter (plus 0.1 s for the speed smoothing)
  cruiseOnly(false);
  const est = arg1 || 'pf', cond = arg2 || 'tailwind';
  for (const fc of [0.1, 0.2, 0.4, 0.8, 1.5])
    for (const seed of SEEDS) {
      const cfg = mk(cond, seed); cfg.estimator = est; cfg.source = 'dsp'; cfg.fc = fc;
      cfg.tune = { posLag: Math.SQRT2 / (2 * Math.PI * fc) + 0.1 };
      const sim = new Sim(cfg);
      for (let i = 0; i < 45 * 6000 && !sim.done; i++) sim.step();
      const m = sim.m;
      out.push({ est, cond, name: `B fc=${fc} lag-compensated`, seed, arrived: sim.done, t: sim.t, speed: (10000 - sim.guidance.dist) / sim.t, wh: m.motorWh, used: m.used,
        rejected: m.rejected + m.abandoned, ctr: m.ctrN ? m.ctrSum / m.ctrN : null });
    }
  save(`rev2_lag_${est}_${cond}`, out);
} else if (mode === 'closed') {
  cruiseOnly(false);
  const est = arg1 || 'pf', cond = arg2 || 'tailwind', only = process.argv[5] || '';
  const cfgs = [0.1, 0.2, 0.4, 0.8, 1.5].map((fc) => ({ name: `B fc=${fc}`, o: { source: 'dsp', fc } }));
  for (const [name, src] of [['C kalman', 'kf'], ['C5 drift-rate Kalman', 'kf5'], ['C+lag Kalman', 'kfl'], ['C+drift+lag Kalman', 'kf5l'], ['C scheduled Kalman', 'kfs'], ['D energy-rate Kalman', 'en'], ['D2 energy-rate Kalman + accelerometer', 'enA'], ['E complementary', 'comp']]) cfgs.push({ name, o: { source: src } });
  if (only) cfgs.length = 0;
  if (only) cfgs.push({ name: `only ${only}`, o: { source: only } });
  for (const c of cfgs)
    for (const seed of SEEDS) {
      const cfg = mk(cond, seed); cfg.estimator = est; Object.assign(cfg, c.o);
      const sim = new Sim(cfg);
      for (let i = 0; i < 45 * 6000 && !sim.done; i++) sim.step();
      const m = sim.m;
      out.push({ est, cond, name: c.name, seed, arrived: sim.done, t: sim.t, speed: (10000 - sim.guidance.dist) / sim.t, wh: m.motorWh, used: m.used,
        rejected: m.rejected + m.abandoned, ctr: m.ctrN ? m.ctrSum / m.ctrN : null, cov: m.coverN ? m.coverIn / m.coverN : null });
    }
  save(`rev2_closed_${est}_${cond}${only ? "_only_" + only : ""}`, out);
} else throw new Error('unknown mode');
console.log('done', mode, arg1, arg2, SUF, out.length);
