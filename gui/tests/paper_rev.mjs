// Revision experiments.   node tests/paper_rev.mjs <mode> [part]
//   latency  fc|gnss|pitot : closed-loop effect of filter cutoff, GNSS delay and pitot lag on the 10 km mission
//   roc                    : detector operating curves; tune seeds 500-529, test seeds 1000-1119, three turbulence levels
//   kalman   on|off        : Kalman ablation (GNSS altitude on/off x barometer-offset noise x airspeed jerk density), with and without barometer drift
//   stress                 : coloured barometer noise, IMU vibration, drag-polar error: pipeline error and detector behaviour
//   signals                : pipeline-B intermediate signals for offline comparison of further filters
import { Sim, defaultConfig, LIMITS } from '../js/sim/sim.js';
import { Cusum } from '../js/sim/estimation.js';
import { RATES, NOISE } from '../js/sim/sensors.js';
import fs from 'node:fs';

const mode = process.argv[2];
const part = process.argv[3] || '';
// REAL=1: sensor rates and noise set to the medians recorded in the ALFA flights (paper/real_data_check.py);
// the Kalman measurement noise follows NOISE, so the filters are retuned to the recorded noise.
const REAL = process.env.REAL === '1';
if (REAL) { RATES.baro = 10; RATES.pitot = 10; RATES.imu = 50; NOISE.baro = 0.25; NOISE.pitot = 0.2; NOISE.accel = 0.26; }
const SUF = REAL ? '_real' : '';
const DTB = 1 / RATES.baro;
const SEEDS = Array.from({ length: 30 }, (_, i) => 100 + i);
const OUT = '../paper/results/';
const CEIL = { ceiling: LIMITS.ceiling, reentry: LIMITS.reentry };
const cruiseOnly = (on) => { LIMITS.ceiling = on ? 0 : CEIL.ceiling; LIMITS.reentry = on ? -1 : CEIL.reentry; };
const save = (name, data) => fs.writeFileSync(OUT + name.replace('.json', SUF + '.json'), JSON.stringify(data));

// ------------------------------------------------------------------ detectors (same definitions as the paper)
function threshold(level) {
  return { n: 0, lock: 0, step(w, dt) { if (this.lock > 0) { this.lock -= dt; this.n = 0; return false; } this.n = w > level ? this.n + dt : 0; if (this.n >= 2) { this.lock = 8; this.n = 0; return true; } return false; } };
}
function cusum(adaptive, h0) {
  const c = new Cusum(0.5, h0);
  return { lock: 0, step(w, dt) {
    if (!adaptive) c.gust = 0;
    const hit = c.step(w, dt, this.lock <= 0);
    if (this.lock > 0) this.lock -= dt;
    if (hit) { this.lock = 8; c.reset(); return true; }
    return false;
  } };
}
const H = [0.75, 1, 1.5, 2, 3, 4, 6, 8];
const detectorBank = () => {
  const d = {};
  for (const l of [0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3]) d[`thr|${l}`] = threshold(l);
  for (const h of H) { d[`fix|${h}`] = cusum(false, h); d[`ada|${h}`] = cusum(true, h); }
  return d;
};

/** One 300 s straight flight through the thermals; every detector sees the same Kalman-filtered signal. */
function detectionFlight(seed, turb, cfgExtra = {}, only = null) {
  const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'oracle'; Object.assign(cfg, cfgExtra);
  const sim = new Sim(cfg);
  sim.atm.turb = turb;
  let dets = detectorBank();
  if (only) dets = Object.fromEntries(Object.entries(dets).filter(([k]) => only.includes(k)));
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

const rms = (sumsq, n) => Math.sqrt(sumsq / n);

// ------------------------------------------------------------------ modes
const out = [];
if (mode === 'latency') {
  const groups = {
    fc: [0.1, 0.2, 0.4, 0.8, 1.5].map((fc) => ({ name: `B fc=${fc}`, kind: 'fc', value: fc, cfg: { source: 'dsp', fc } })),
    gnss: [{ name: 'baseline', kind: 'base', value: 0.15, cfg: {} },
      ...[0.5, 1, 2].map((d) => ({ name: `gnss ${d}`, kind: 'gnss', value: d, cfg: { gnssDelay: d } })),
      ...[0.5, 1, 2].map((d) => ({ name: `gnss ${d} uncompensated`, kind: 'gnssU', value: d, cfg: { gnssDelay: d, gnssComp: false } }))],
    pitot: [0.5, 1, 2].map((p) => ({ name: `pitot ${p}`, kind: 'pitot', value: p, cfg: { pitotTau: p } })),
  }[part];
  for (const c of groups)
    for (const seed of SEEDS) {
      const cfg = defaultConfig('tailwind'); cfg.estimator = 'pf'; cfg.seed = seed; Object.assign(cfg, c.cfg);
      const sim = new Sim(cfg);
      for (let i = 0; i < 45 * 6000 && !sim.done; i++) sim.step();
      const m = sim.m;
      out.push({ name: c.name, kind: c.kind, value: c.value, seed, arrived: sim.done, t: sim.t, speed: (10000 - sim.guidance.dist) / sim.t, wh: m.motorWh, used: m.used, rejected: m.rejected + m.abandoned,
        gain: m.gain, climbTime: m.climbTime, acc: m.accN ? m.accSum / m.accN : null, ctr: m.ctrN ? m.ctrSum / m.ctrN : null });
    }
  save(`rev_latency_${part}.json`, out);
} else if (mode === 'roc') {
  cruiseOnly(true);
  const plan = [['tune', Array.from({ length: 30 }, (_, i) => 500 + i)], ['test', Array.from({ length: 120 }, (_, i) => 1000 + i)]];
  for (const [split, seeds] of plan)
    for (const turb of [0.2, 0.5, 0.9])
      for (const seed of seeds) out.push({ split, ...detectionFlight(seed, turb) });
  save('rev_roc.json', out);
} else if (mode === 'kalman') {
  const gnss = part === 'on';
  for (const baro of [false, true])
    for (const qb of [0.01, 0.05, 0.5])
      for (const qj of [0.25, 1, 4])
        for (const seed of SEEDS) {
          const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'pf'; cfg.tune = { kf: { qb, qj, gnss } };
          if (baro) cfg.faults.baro = true;
          const sim = new Sim(cfg);
          const a = { s: [0, 0], m: [0, 0] };
          let last = 0, vPrev = sim.glider.V;
          for (let i = 0; i < 300 * 100 && !sim.done; i++) {
            sim.step();
            if (sim.vt === last) continue;
            const vd = (sim.glider.V - vPrev) / DTB; vPrev = sim.glider.V; last = sim.vt;
            if (sim.t < 10) continue;
            const k = Math.abs(vd) < 0.15 ? a.s : a.m;
            k[0] += (sim.chain.wKf - sim.air.w) ** 2; k[1]++;
          }
          out.push({ gnss, qb, qj, baro, seed, steady: rms(a.s[0], a.s[1]), manoeuvre: rms(a.m[0], a.m[1]), ns: a.s[1], nm: a.m[1] });
        }
  save(`rev_kalman_${part}.json`, out);
} else if (mode === 'stress') {
  const conds = {
    nominal: {},
    'coloured baro noise': { stress: { tauBaro: 1 } },
    'IMU vibration 30 Hz': { stress: { vibAmp: 0.5, vibF: 30 } },
    'IMU vibration 3 Hz': { stress: { vibAmp: 1.0, vibF: 3 } },
    'polar -20%': { polarErr: -0.2 },
    'polar +20%': { polarErr: 0.2 },
    'all combined': { stress: { tauBaro: 1, vibAmp: 0.5, vibF: 30 }, polarErr: 0.2 },
  };
  for (const [name, c] of Object.entries(conds)) {
    for (const seed of SEEDS) {
      cruiseOnly(false);
      const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'oracle'; Object.assign(cfg, c);
      const sim = new Sim(cfg);
      const acc = { A: [0, 0], B: [0, 0], C: [0, 0], n: 0 };
      let last = 0;
      for (let i = 0; i < 300 * 100 && !sim.done; i++) {
        sim.step();
        if (sim.vt !== last && sim.t > 15) {
          last = sim.vt;
          const w = sim.air.w, ch = sim.chain;
          for (const [k, v] of [['A', ch.wRaw], ['B', ch.wDsp], ['C', ch.wKf]]) { acc[k][0] += (v - w) ** 2; acc[k][1] += v - w; }
          acc.n++;
        }
      }
      cruiseOnly(true);
      const det = detectionFlight(seed, 0.5, c, ['thr|1', 'fix|1.5', 'ada|1.5']);
      out.push({ cond: name, seed, rmse: { A: rms(acc.A[0], acc.n), B: rms(acc.B[0], acc.n), C: rms(acc.C[0], acc.n) }, bias: { A: acc.A[1] / acc.n, B: acc.B[1] / acc.n, C: acc.C[1] / acc.n }, det });
    }
  }
  save('rev_stress.json', out);
} else if (mode === 'signals') {
  cruiseOnly(true);
  for (const seed of SEEDS) {
    const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'oracle';
    const sim = new Sim(cfg);
    const r = { seed, w: [], eh: [], ed: [], sink: [] };
    let last = 0;
    for (let i = 0; i < 240 * 100; i++) {
      sim.step();
      if (sim.vt === last) continue;
      last = sim.vt;
      const d = sim.chain.dbg;
      r.w.push(+sim.air.w.toFixed(4)); r.eh.push(+d.eh.toFixed(4)); r.ed.push(+d.ed.toFixed(4)); r.sink.push(+d.sink.toFixed(4));
    }
    out.push(r);
  }
  save('rev_signals.json', out);
} else throw new Error('unknown mode');
console.log('done', mode, part, out.length);
