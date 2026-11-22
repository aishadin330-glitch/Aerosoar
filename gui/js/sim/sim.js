// Simulation orchestrator: truth world → sensors → estimation → guidance → control, at 100 Hz.
// Truth is used in exactly two places: stepping the world, and scoring (metrics / oracle baseline).
import { RNG, Ring, clamp } from './util.js';
import { AC, Glider, Atmosphere } from './physics.js';
import { Sensors, RATES } from './sensors.js';
import { VarioChain, Cusum, ThermalPF, ThermalEKF } from './estimation.js';
import { Guidance, STATES, LIMITS } from './guidance.js';

export const DT = 0.01;

/** Learning-model input: the last WIN samples (2 Hz) of FEAT channels, positions relative to the current estimate. */
export const WIN = 32;
export const FEAT = ['px', 'py', 'w', 'sig', 'ch', 'sh', 'v', 'wu', 'wv'];

/** Point-to-point mission: start at the origin, destination 10 km due east. */
export const ROUTE = { length: 10000, h0: 300 };

export const SCENARIOS = {
  calm: {
    name: 'Calm',
    note: 'No wind. Thermals of constant strength that conform to the Gaussian estimator model.',
    wind: 0, from: 270, turb: 0.2, sinkRing: false, life: 'steady', w: [3.0, 4.2], extend: -500,
  },
  tailwind: {
    name: 'Tailwind',
    note: 'Tailwind of 4 m/s. Thermals drift along the route and vary periodically in strength.',
    wind: 4, from: 270, turb: 0.45, sinkRing: false, life: 'pulse', w: [2.8, 4.0], extend: -500,
  },
  adverse: {
    name: 'Adverse',
    note: 'Headwind of 3 m/s, strong turbulence, and thermals surrounded by a sink ring (outside the estimator model).',
    wind: 3, from: 90, turb: 0.85, sinkRing: true, life: 'pulse', w: [2.6, 3.6], extend: 3000,
  },
};

/** Thermals are placed close to the direct track so that a straight course encounters them. */
function layThermals(atm, rng, sc) {
  for (let x = 700; x < ROUTE.length + sc.extend; x += rng.uni(750, 1050))
    atm.addThermal({ x, y: clamp(rng.gauss(40), -85, 85), wmax: rng.uni(sc.w[0], sc.w[1]), sigma: rng.uni(55, 80), life: sc.life });
}

export function defaultConfig(scenario = 'tailwind') {
  return {
    scenario,
    seed: 7,
    noise: 1, // multiplier on all sensor noise standard deviations
    faults: { baro: false, gnss: false, pitot: false },
    estimator: 'pf', // 'pf' | 'ekf' | 'ml' | 'reactive' | 'oracle'
    // fixed in the interface; retained as parameters for batch experiments
    polarErr: 0,
    filter: 'butter',
    fc: 0.4,
    source: 'kf', // vertical-air-velocity pipeline used for detection and estimation: 'kf' | 'dsp'
  };
}

export class Sim {
  constructor(cfg) {
    this.cfg = cfg;
    this.goal = { x: ROUTE.length, y: 0 };
    this.hist = new Ring(1200, ['t', 'wTrue', 'wDsp', 'wKf', 'sig']); // 20 Hz, 60 s
    this.trail = new Ring(6000, ['x', 'y', 'h', 'w']); // 4 Hz, 25 min
    this.feat = new Ring(WIN, FEAT); // 2 Hz, 16 s
    this.mlHist = new Ring(240, ['t', 'err', 'errEst', 'p']); // 2 Hz, 2 min
    this.mlModel = null; // set by the interface once the weights are loaded
    this.reset();
  }
  reset() {
    const cfg = this.cfg;
    const sc = cfg.custom || SCENARIOS[cfg.scenario];
    this.atm = new Atmosphere(new RNG(cfg.seed * 104729 + 1));
    this.atm.sinkRing = sc.sinkRing;
    this.atm.turb = sc.turb;
    if (cfg.hGust) this.atm.hGust = cfg.hGust;
    this.atm.setWind(sc.wind, sc.from);
    if (sc.thermals) for (const t of sc.thermals) this.atm.addThermal(t);
    else layThermals(this.atm, this.atm.rng, sc);
    this.glider = new Glider();
    this.glider.reset(0, 0, ROUTE.h0, 0);
    this.sensors = new Sensors(new RNG(cfg.seed * 31337 + 5));
    this.chain = this.chain || new VarioChain();
    this.chain.setFilter(cfg.filter, cfg.fc);
    this.chain.reset(this.glider);
    this.chain.configure({ ...(cfg.tune?.kf || {}), comp: cfg.gnssComp });
    this.cusum = new Cusum();
    this.guidance = new Guidance(new RNG(cfg.seed * 271 + 9), this.goal);
    this.est = null;
    this.trackActive = false;
    this.done = false;
    this.t = 0;
    this.k = 0;
    this.vt = 0;
    this.w = 0;
    this.sig = 1;
    this.wDot = 0;
    this.wPrev = 0;
    this.trip = false;
    this.air = { u: 0, v: 0, w: 0, wT: 0 };
    this.truthIn = false;
    this.tEnter = 0;
    this.climbEntryH = null;
    this.hist.clear();
    this.trail.clear();
    this.feat.clear();
    this.mlHist.clear();
    this.ml = null;
    this.m = {
      rmsDsp: 0, rmsKf: 0, // exponentially windowed mean-square error (≈30 s)
      detections: 0, falseAlarms: 0, latSum: 0, latN: 0,
      ctrErr: null, coverIn: 0, coverN: 0, ctrSum: 0, ctrN: 0,
      gain: 0, used: 0, abandoned: 0, rejected: 0, motorWh: 0, climbTime: 0, accSum: 0, accN: 0,
    };
  }
  setEstimator() {
    // switching estimator during a climb: drop the track and allow re-detection
    if (this.trackActive) this.guidance.resume(this.ctx(), 'Estimator changed; course resumed');
  }
  makeEstimator() {
    const kind = this.cfg.estimator;
    if (kind === 'pf') return new ThermalPF(new RNG(this.cfg.seed * 977 + this.k), 400, this.cfg.tune?.q);
    if (kind === 'ekf') return new ThermalEKF();
    return null;
  }
  /** Oracle baseline: reads truth. Simulation upper bound only; not deployable. */
  oracleEstimate() {
    const g = this.glider;
    const th = this.atm.dominant(g.x, g.y, 320);
    if (!th) return null;
    return { x: th.x, y: th.y, w: th.wmax * th.s * this.atm.hprof(g.h), s: th.sigma, Pxx: 0, Pxy: 0, Pyy: 0, sdW: 0, sdS: 0, posSd: 0 };
  }
  /** Latest network output as an estimate, advected with the wind since it was computed. */
  mlEstimate() {
    const o = this.ml;
    if (!o || o.p < 0.3) return null;
    const dt = this.t - o.t;
    const n = this.chain.nav;
    const Pxx = o.sdx ** 2;
    const Pyy = o.sdy ** 2;
    return { x: o.x + n.wu * dt, y: o.y + n.wv * dt, w: o.w, s: o.s, Pxx, Pxy: 0, Pyy, sdW: o.sdw, sdS: o.sds, posSd: Math.sqrt((Pxx + Pyy) / 2) };
  }
  /** Run the network on the current window (2 Hz) and score it against truth. */
  runML() {
    const m = this.mlModel;
    const win = m && this.window();
    if (!win) return;
    const t0 = performance.now();
    const o = m.predict(win);
    const n = this.chain.nav;
    this.ml = { ...o, x: n.x + o.cx, y: n.y + o.cy, t: this.t, ms: performance.now() - t0 };
    const g = this.glider;
    const th = this.atm.dominant(g.x, g.y, 250);
    const weff = th ? th.wmax * th.s * this.atm.hprof(g.h) : 0;
    const real = th && weff > 0.5;
    const e = this.cfg.estimator !== 'ml' && this.trackActive ? this.estimate() : null;
    this.mlHist.push({
      t: this.t, p: o.p,
      err: real ? Math.hypot(this.ml.x - th.x, this.ml.y - th.y) : NaN,
      errEst: real && e ? Math.hypot(e.x - th.x, e.y - th.y) : NaN,
    });
  }
  estimate() {
    if (!this.trackActive) return null;
    if (this.cfg.estimator === 'oracle') return this.oracleEstimate();
    if (this.cfg.estimator === 'ml') return this.mlEstimate();
    return this.est ? this.est.get() : null;
  }
  /** Model input for the current moment (WIN×FEAT, row-major), or null until the window is full. */
  window() {
    const F = this.feat;
    if (F.len < WIN) return null;
    const out = new Float32Array(WIN * FEAT.length);
    const j0 = F.idx(WIN - 1);
    for (let i = 0; i < WIN; i++) {
      const j = F.idx(i);
      const o = i * FEAT.length;
      out[o] = (F.px[j] - F.px[j0]) / 100;
      out[o + 1] = (F.py[j] - F.py[j0]) / 100;
      out[o + 2] = F.w[j] / 2;
      out[o + 3] = F.sig[j];
      out[o + 4] = F.ch[j];
      out[o + 5] = F.sh[j];
      out[o + 6] = F.v[j] / 14 - 1;
      out[o + 7] = F.wu[j] / 5;
      out[o + 8] = F.wv[j] / 5;
    }
    return out;
  }
  /** Everything the guidance logic is permitted to see: estimates only. */
  ctx() {
    const n = this.chain.nav;
    const v = this.chain.vkf;
    return {
      t: this.t, x: n.x, y: n.y, h: v.h, vz: v.vz, chi: n.chi, V: n.V, wu: n.wu, wv: n.wv,
      w: this.w, sig: this.sig, wDot: this.wDot, trip: this.trip,
      est: this.estimate(), kind: this.cfg.estimator,
      startTrack: () => this.startTrack(),
      stopTrack: () => this.stopTrack(),
      emit: (type, d) => this.onEvent(type, d),
    };
  }
  startTrack() {
    const n = this.chain.nav;
    this.trackActive = true;
    this.est = this.makeEstimator();
    if (this.est) this.est.init(n.x + 25 * Math.cos(n.chi), n.y + 25 * Math.sin(n.chi), Math.max(0.6, this.w));
    // score the detection against truth
    const m = this.m;
    m.detections++;
    if (this.air.wT < 0.3) m.falseAlarms++;
    else if (this.truthIn && this.t - this.tEnter < 15) {
      m.latSum += this.t - this.tEnter;
      m.latN++;
    }
  }
  stopTrack() {
    this.trackActive = false;
    this.est = null;
  }
  onEvent(type, d) {
    if (type !== 'state') return;
    const g = this.glider;
    if (d.to === 'ASSESS') this.tAssess = this.t;
    if (d.to === 'CLIMB') {
      this.climbEntryH = g.h;
      if (d.from === 'ASSESS') { this.m.accSum += this.t - this.tAssess; this.m.accN++; }
    }
    if (d.from === 'CLIMB' && this.climbEntryH !== null) {
      if (g.h - this.climbEntryH > 30) this.m.used++;
      else this.m.abandoned++;
      this.climbEntryH = null;
    }
    if (d.from === 'ASSESS' && d.to === 'CRUISE') this.m.rejected++;
    if (d.to === 'ARRIVED') this.done = true;
  }

  step() {
    if (this.done) return;
    const cfg = this.cfg;
    const g = this.glider;
    const atm = this.atm;
    this.t += DT;
    this.k++;
    atm.step(DT);
    this.air = atm.sample(g.x, g.y, g.h);
    g.step(DT, this.air);

    const s = this.sensors.step(this.t, DT, g, cfg);
    const ch = this.chain;
    if (s.imu) ch.onImu(s.imu, 1 / RATES.imu);
    if (s.ahrs) ch.onAhrs(s.ahrs);
    if (s.pitot) ch.onPitot(s.pitot);
    ch.nav.predict(DT);
    if (s.gnss) ch.onGnss(s.gnss, this.t);
    if (s.baro) {
      ch.onBaro(s.baro, g.motor, cfg.polarErr);
      this.varioTick(1 / RATES.baro);
    }
    if (g.motor) this.m.motorWh += (AC.motorP * DT) / 3600;
  }

  varioTick(dt) {
    const cfg = this.cfg;
    const ch = this.chain;
    const g = this.glider;
    const m = this.m;
    const gd = this.guidance;
    const atm = this.atm;
    this.vt++;

    const SRC = { dsp: ch.wDsp, kf: ch.wKf, kf5: ch.wKf5, kfl: ch.wKfL, kf5l: ch.wKf5L, kfs: ch.wKfS, kf5s: ch.wKf5S, ki: ch.wKfI, en: ch.wEn, enA: ch.wEnA, comp: ch.wComp };
    this.w = SRC[cfg.source] ?? ch.wKf;
    this.sig = cfg.source === undefined || ['kf', 'kf5', 'kfl', 'kf5l', 'kfs', 'kf5s', 'ki'].includes(cfg.source) ? ch.sigKf : 0.45;
    this.wDot += (((this.w - this.wPrev) / dt - this.wDot) * dt) / 1.5;
    this.wPrev = this.w;

    // the detector can alarm only while cruising
    this.trip = this.cusum.step(this.w, dt, gd.state === 'CRUISE');

    // truth bookkeeping for detection latency (scoring only)
    const inLift = this.air.wT > 0.6;
    if (inLift && !this.truthIn) this.tEnter = this.t;
    this.truthIn = inLift;

    // delay-compensated assignment (cfg.tune.posLag, seconds): a w estimate that lags the air by tau belongs to the position travelled tau earlier
    const lagC = cfg.tune?.posLag ?? 0;
    if (lagC > 0) {
      (this.posHist ??= []).push([this.t, ch.nav.x, ch.nav.y]);
      while (this.posHist.length > 4 && this.posHist[0][0] < this.t - 10) this.posHist.shift();
    }
    // thermal estimator at 10 Hz
    if (this.trackActive && this.est && this.vt % 2 === 0) {
      const n = ch.nav;
      let px = n.x, py = n.y;
      if (lagC > 0) {
        const h = this.posHist, tt = this.t - lagC;
        let i = h.length - 1;
        while (i > 0 && h[i][0] > tt) i--;
        if (h[i][0] <= tt && i + 1 < h.length) {
          const f = (tt - h[i][0]) / (h[i + 1][0] - h[i][0]);
          px = h[i][1] + f * (h[i + 1][1] - h[i][1]);
          py = h[i][2] + f * (h[i + 1][2] - h[i][2]);
        } else { px = h[i][1]; py = h[i][2]; }
      }
      // 10 Hz samples share filter and gust errors, so R is inflated (×10) rather than treated as independent
      this.est.predict(0.1, n.wu, n.wv);
      this.est.update(px, py, this.w, (cfg.tune?.rScale ?? 10) * (this.sig * this.sig + 0.3 * 0.3));
    }

    const cmd = gd.update(dt, this.ctx());
    g.phiCmd = cmd.phi;
    g.Vcmd = cfg.speedDoublet ? cmd.V + cfg.speedDoublet * Math.sign(Math.sin((2 * Math.PI * this.t) / 40)) : cmd.V;
    g.motor = cmd.motor;

    // ---- scoring against truth ----
    const wTrue = this.air.w;
    if (this.t > 3) {
      const a = dt / 30;
      m.rmsDsp += a * ((ch.wDsp - wTrue) ** 2 - m.rmsDsp);
      m.rmsKf += a * ((ch.wKf - wTrue) ** 2 - m.rmsKf);
    }
    const e = this.estimate();
    const th = e ? atm.dominant(g.x, g.y, 320) : null;
    if (th) {
      const dx = th.x - e.x;
      const dy = th.y - e.y;
      m.ctrErr = Math.hypot(dx, dy);
      if (gd.state === 'CLIMB') {
        m.ctrSum += m.ctrErr;
        m.ctrN++;
        const det = e.Pxx * e.Pyy - e.Pxy * e.Pxy;
        if (det > 1e-6) {
          const d2 = (e.Pyy * dx * dx - 2 * e.Pxy * dx * dy + e.Pxx * dy * dy) / det;
          m.coverN++;
          if (d2 < 5.99) m.coverIn++; // inside the 95 % region
        }
      }
    } else m.ctrErr = null;
    if (gd.state === 'CLIMB') m.climbTime += dt;
    if (!g.motor && g.hdot > 0 && gd.state === 'CLIMB') m.gain += g.hdot * dt;

    this.hist.push({ t: this.t, wTrue, wDsp: ch.wDsp, wKf: ch.wKf, sig: ch.sigKf });
    if (this.vt % 10 === 0) {
      const n = ch.nav;
      this.feat.push({ px: n.x, py: n.y, w: this.w, sig: this.sig, ch: Math.cos(n.chi), sh: Math.sin(n.chi), v: n.V, wu: n.wu, wv: n.wv });
      this.runML();
    }
    if (this.vt % 5 === 0) this.trail.push({ x: g.x, y: g.y, h: g.h, w: wTrue });
  }
}

export { STATES, LIMITS, AC };
