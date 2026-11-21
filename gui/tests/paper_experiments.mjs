// Experiments for the paper.  node tests/paper_experiments.mjs missions <scenario> | pipelines
// missions: 6 estimator configurations x 30 seeds on the 10 km mission.  pipelines: vertical-air-velocity error under faults.
import { Sim, defaultConfig } from '../js/sim/sim.js';
import { MLModel } from '../js/sim/ml.js';
import fs from 'node:fs';

const mode = process.argv[2];
const model = new MLModel(JSON.parse(fs.readFileSync('ml/transformer.json')));
const SEEDS = Array.from({ length: 30 }, (_, i) => 100 + i);
const out = [];

if (mode === 'missions') {
  const scenario = process.argv[3];
  const CONFIGS = {
    oracle: { estimator: 'oracle' }, pf: { estimator: 'pf' }, ekf: { estimator: 'ekf' },
    ml: { estimator: 'ml' }, reactive: { estimator: 'reactive' }, 'pf-dsp': { estimator: 'pf', source: 'dsp' },
  };
  for (const [name, c] of Object.entries(CONFIGS))
    for (const seed of SEEDS) {
      const cfg = defaultConfig(scenario); Object.assign(cfg, c); cfg.seed = seed;
      const sim = new Sim(cfg); sim.mlModel = model;
      for (let i = 0; i < 45 * 6000 && !sim.done; i++) sim.step();
      const m = sim.m;
      out.push({ scenario, est: name, seed, arrived: sim.done, t: sim.t, speed: (10000 - sim.guidance.dist) / sim.t, used: m.used, rejected: m.rejected + m.abandoned,
        wh: m.motorWh, ctr: m.ctrN ? m.ctrSum / m.ctrN : null, cov: m.coverN ? m.coverIn / m.coverN : null, det: m.detections, fa: m.falseAlarms });
    }
  fs.writeFileSync(`../paper/results/missions_${scenario}.json`, JSON.stringify(out));
} else {
  const CONDS = { nominal: {}, 'baro drift': { faults: { baro: true } }, 'GNSS outage': { faults: { gnss: true } }, 'pitot error': { faults: { pitot: true } }, 'noise x3': { noise: 3 } };
  for (const [cond, c] of Object.entries(CONDS))
    for (const seed of SEEDS) {
      const cfg = defaultConfig('tailwind'); cfg.seed = seed; cfg.estimator = 'oracle';
      Object.assign(cfg, { noise: c.noise ?? 1 }); cfg.faults = { baro: false, gnss: false, pitot: false, ...(c.faults || {}) };
      const sim = new Sim(cfg);
      const acc = { A: [0, 0], B: [0, 0], C: [0, 0], n: 0 };
      let last = 0;
      for (let i = 0; i < 300 * 100 && !sim.done; i++) {
        sim.step();
        if (sim.vt !== last && sim.t > 15) {
          last = sim.vt; const w = sim.air.w, ch = sim.chain;
          for (const [k, v] of [['A', ch.wRaw], ['B', ch.wDsp], ['C', ch.wKf]]) { acc[k][0] += (v - w) ** 2; acc[k][1] += v - w; }
          acc.n++;
        }
      }
      out.push({ cond, seed, rmse: { A: Math.sqrt(acc.A[0] / acc.n), B: Math.sqrt(acc.B[0] / acc.n), C: Math.sqrt(acc.C[0] / acc.n) }, bias: { A: acc.A[1] / acc.n, B: acc.B[1] / acc.n, C: acc.C[1] / acc.n } });
    }
  fs.writeFileSync('../paper/results/pipelines.json', JSON.stringify(out));
}
console.log('done', mode, process.argv[3] || '', out.length);
