// Batch check of the simulation core without a browser:  node tests/headless.mjs [maxMinutes]
// Flies the 10 km mission for every scenario and estimator over three seeds.
import { Sim, defaultConfig, SCENARIOS, ROUTE } from '../js/sim/sim.js';

const maxMinutes = Number(process.argv[2] || 40);
const rows = [];
for (const scenario of Object.keys(SCENARIOS))
  for (const estimator of ['oracle', 'pf', 'ekf', 'reactive']) {
    const a = { arrived: 0, time: 0, speed: 0, gain: 0, used: 0, aband: 0, rej: 0, wh: 0, fa: 0, det: 0, ctr: 0, ctrN: 0, cov: 0, covN: 0, kf: 0, dsp: 0, n: 0, bad: 0 };
    for (const seed of [3, 7, 11]) {
      const cfg = defaultConfig(scenario);
      cfg.estimator = estimator;
      cfg.seed = seed;
      const sim = new Sim(cfg);
      for (let i = 0; i < maxMinutes * 6000 && !sim.done; i++) sim.step();
      const m = sim.m;
      const g = sim.glider;
      if (![g.x, g.y, g.h, g.V, sim.chain.wKf].every(Number.isFinite)) a.bad++;
      if (sim.done) a.arrived++;
      a.time += sim.t / 60;
      a.speed += (ROUTE.length - sim.guidance.dist) / sim.t;
      a.gain += m.gain; a.used += m.used; a.aband += m.abandoned; a.rej += m.rejected; a.wh += m.motorWh;
      a.fa += m.falseAlarms; a.det += m.detections; a.ctr += m.ctrSum; a.ctrN += m.ctrN; a.cov += m.coverIn; a.covN += m.coverN;
      a.kf += Math.sqrt(m.rmsKf); a.dsp += Math.sqrt(m.rmsDsp); a.n++;
    }
    rows.push({
      scenario, estimator, arrived: `${a.arrived}/${a.n}`, 'time min': (a.time / a.n).toFixed(1), 'task speed m/s': (a.speed / a.n).toFixed(1),
      'gain m': (a.gain / a.n).toFixed(0), 'used/aband/rej': `${a.used}/${a.aband}/${a.rej}`, 'motor Wh': (a.wh / a.n).toFixed(1),
      'det/false': `${a.det}/${a.fa}`, 'ctr err m': a.ctrN ? (a.ctr / a.ctrN).toFixed(1) : '-', 'cover95 %': a.covN ? ((100 * a.cov) / a.covN).toFixed(0) : '-',
      'rmse dsp/kf': `${(a.dsp / a.n).toFixed(2)}/${(a.kf / a.n).toFixed(2)}`, nonfinite: a.bad,
    });
  }
console.table(rows);
