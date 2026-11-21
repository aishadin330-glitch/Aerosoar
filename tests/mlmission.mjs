// Mission comparison including the trained network:  node tests/mlmission.mjs
import { Sim, defaultConfig, SCENARIOS } from '../js/sim/sim.js';
import { MLModel } from '../js/sim/ml.js';
import fs from 'node:fs';
const model = new MLModel(JSON.parse(fs.readFileSync('ml/transformer.json')));
const rows = [];
for (const scenario of Object.keys(SCENARIOS))
  for (const estimator of ['pf', 'ml', 'oracle']) {
    const a = { arr: 0, t: 0, used: 0, rej: 0, wh: 0, n: 0, ctr: 0, ctrN: 0 };
    for (const seed of [3, 7, 11, 13, 17]) {
      const cfg = defaultConfig(scenario); cfg.estimator = estimator; cfg.seed = seed;
      const sim = new Sim(cfg); sim.mlModel = model;
      for (let i = 0; i < 45 * 6000 && !sim.done; i++) sim.step();
      a.arr += sim.done ? 1 : 0; a.t += sim.t / 60; a.used += sim.m.used; a.rej += sim.m.rejected + sim.m.abandoned; a.wh += sim.m.motorWh; a.n++; a.ctr += sim.m.ctrSum; a.ctrN += sim.m.ctrN;
    }
    rows.push({ scenario, estimator, arrived: `${a.arr}/${a.n}`, 'time min': (a.t / a.n).toFixed(1), 'used/rejected': `${a.used}/${a.rej}`, 'motor Wh': (a.wh / a.n).toFixed(1), 'ctr err m': a.ctrN ? (a.ctr / a.ctrN).toFixed(0) : '-' });
  }
console.table(rows);
