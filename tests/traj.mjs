// Export one mission trajectory for the paper figure.  node tests/traj.mjs
import { Sim, defaultConfig } from '../js/sim/sim.js';
import fs from 'node:fs';

const cfg = defaultConfig('tailwind');
cfg.seed = 7;
cfg.estimator = 'pf';
const sim = new Sim(cfg);
const pts = [];
for (let i = 0; i < 45 * 6000 && !sim.done; i++) {
  sim.step();
  if (sim.vt % 10 === 0 && sim.k % 5 === 0) pts.push([sim.t, sim.glider.x, sim.glider.y, sim.glider.h, sim.air.w, sim.guidance.state]);
}
fs.writeFileSync('../paper/results/traj.json', JSON.stringify({ pts, tEnd: sim.t, arrived: sim.done }));
console.log('traj', pts.length, 'arrived', sim.done, 't', sim.t.toFixed(0));
