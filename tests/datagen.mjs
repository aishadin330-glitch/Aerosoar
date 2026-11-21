// Generate labelled windows for the learning models:  node tests/datagen.mjs <split> <firstSeed> <count>
// split: train | val | test | ood.  Output: ../ml/data/<split>_<firstSeed>.{x,y}.f32
import { Sim, defaultConfig, WIN, FEAT, ROUTE } from '../js/sim/sim.js';
import { RNG } from '../js/sim/util.js';
import fs from 'node:fs';

const [split, first, count] = [process.argv[2], Number(process.argv[3]), Number(process.argv[4])];
const OOD = split === 'ood';
const X = [];
const Y = [];

function custom(rng) {
  const th = [];
  for (let x = 500; x < ROUTE.length; x += rng.uni(550, 950))
    th.push({
      x, y: rng.gauss(60),
      wmax: OOD ? rng.uni(1.2, 2.2) : rng.uni(1.5, 4.6),
      sigma: OOD ? rng.uni(90, 120) : rng.uni(40, 90),
      life: rng.next() < 0.5 ? 'steady' : 'pulse',
    });
  return {
    thermals: th, sinkRing: OOD, turb: OOD ? rng.uni(0.9, 1.3) : rng.uni(0.2, 0.8),
    wind: OOD ? rng.uni(6, 9) : rng.uni(0, 6), from: rng.uni(0, 360),
  };
}

for (let k = 0; k < count; k++) {
  const seed = first + k;
  const rng = new RNG(seed * 7 + 1);
  const cfg = defaultConfig('calm');
  cfg.seed = seed;
  cfg.estimator = 'oracle';
  cfg.custom = custom(rng);
  cfg.noise = OOD ? rng.uni(2.5, 4) : rng.uni(0.5, 2);
  if (!OOD && rng.next() < 0.25) cfg.faults[['baro', 'gnss', 'pitot'][Math.floor(rng.uni(0, 3))]] = true;
  if (OOD) cfg.faults.baro = rng.next() < 0.5;
  const sim = new Sim(cfg);
  sim.atm.setWind(cfg.custom.wind, cfg.custom.from);
  for (let i = 0; i < 20 * 60 * 100 && !sim.done; i++) {
    sim.step();
    if (sim.vt % 20 !== 0 || sim.k % 5 !== 0) continue; // 1 Hz
    const w = sim.window();
    if (!w) continue;
    const n = sim.chain.nav;
    const th = sim.atm.dominant(sim.glider.x, sim.glider.y, 250);
    const weff = th ? th.wmax * th.s * sim.atm.hprof(sim.glider.h) : 0;
    const exist = th && weff > 0.5 ? 1 : 0;
    if (!exist && rng.next() > 0.3) continue; // keep 30 % of negatives
    X.push(w);
    Y.push(Float32Array.of(exist, exist ? (th.x - n.x) / 100 : 0, exist ? (th.y - n.y) / 100 : 0, exist ? weff / 4 : 0, exist ? th.sigma / 60 : 0, seed));
  }
}
const cat = (a) => { const o = new Float32Array(a.reduce((s, v) => s + v.length, 0)); let p = 0; for (const v of a) { o.set(v, p); p += v.length; } return Buffer.from(o.buffer); };
fs.mkdirSync('../ml/data', { recursive: true });
fs.writeFileSync(`../ml/data/${split}_${first}.x.f32`, cat(X));
fs.writeFileSync(`../ml/data/${split}_${first}.y.f32`, cat(Y));
console.log(split, first, 'windows', X.length, 'positives', Y.filter((y) => y[0] === 1).length);
