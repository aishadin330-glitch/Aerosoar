// Onboard estimation (blueprint §7–8). Nothing in this file may read simulator truth:
// inputs are sensor samples only. Three vertical-air-velocity pipelines run side by side,
// then a change detector and a thermal-parameter estimator (particle filter or EKF).
import { G, clamp } from './util.js';
import { AC, dragSink } from './physics.js';
import { makeFilter } from './dsp.js';
import { NOISE, RATES } from './sensors.js';

/** Minimal Kalman filter with scalar measurement updates (no matrix inverse needed). */
class KF {
  constructor(n) {
    this.n = n;
    this.x = new Float64Array(n);
    this.P = new Float64Array(n * n);
  }
  setDiag(d) {
    this.P.fill(0);
    d.forEach((v, i) => (this.P[i * this.n + i] = v));
  }
  /** P ← F P Fᵀ + Q (state propagation is done by the caller). */
  propagate(F, Q) {
    const n = this.n;
    const P = this.P;
    const T = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += F[i * n + k] * P[k * n + j];
        T[i * n + j] = s;
      }
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = Q[i * n + j];
        for (let k = 0; k < n; k++) s += T[i * n + k] * F[j * n + k];
        P[i * n + j] = s;
      }
  }
  /** Scalar update. hx = predicted measurement, H = measurement Jacobian row. Returns normalised innovation². */
  update(z, hx, H, R) {
    const n = this.n;
    const P = this.P;
    const PHt = new Float64Array(n);
    let S = R;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += P[i * n + j] * H[j];
      PHt[i] = s;
      S += H[i] * s;
    }
    const innov = z - hx;
    for (let i = 0; i < n; i++) {
      const K = PHt[i] / S;
      this.x[i] += K * innov;
      for (let j = 0; j < n; j++) P[i * n + j] -= K * PHt[j];
    }
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const a = 0.5 * (P[i * n + j] + P[j * n + i]);
        P[i * n + j] = a;
        P[j * n + i] = a;
      }
    return (innov * innov) / S;
  }
}

/**
 * Vertical channel: state [h, vz, accel bias, baro bias].
 * IMU acceleration drives the prediction; barometer and GNSS altitude correct it.
 * GNSS altitude is what makes the barometer bias (pressure drift) observable.
 */
export class VerticalKF extends KF {
  constructor() {
    super(4);
    this.reset(0);
  }
  reset(h0) {
    this.qb = this.qb ?? 0.05; // random-walk density of the barometer offset
    this.x.set([h0, 0, 0, 0]);
    this.setDiag([9, 4, 0.02, 4]);
  }
  predict(az, dt, extraVar = 0) {
    const x = this.x;
    const a = az - x[2];
    x[0] += x[1] * dt + 0.5 * a * dt * dt;
    x[1] += a * dt;
    const h2 = 0.5 * dt * dt;
    const F = [1, dt, -h2, 0, 0, 1, -dt, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const sa2 = 0.2 * 0.2 + extraVar;
    const Q = new Float64Array(16);
    Q[0] = h2 * h2 * sa2;
    Q[1] = Q[4] = h2 * dt * sa2;
    Q[5] = dt * dt * sa2;
    Q[10] = 0.003 * 0.003 * dt;
    Q[15] = this.qb * this.qb * dt;
    this.propagate(F, Q);
  }
  baro(z) {
    return this.update(z, this.x[0] + this.x[3], [1, 0, 0, 1], (NOISE.baro * 1.3) ** 2);
  }
  gnssAlt(z) {
    return this.update(z, this.x[0], [1, 0, 0, 0], NOISE.gnssH ** 2);
  }
  get h() {
    return this.x[0];
  }
  get vz() {
    return this.x[1];
  }
  get varVz() {
    return this.P[5];
  }
}

/** Vertical channel with an extra barometer drift-rate state: [h, vz, accel bias, baro offset, baro offset rate]. */
export class VerticalKF5 extends KF {
  constructor() {
    super(5);
    this.reset(0);
  }
  reset(h0) {
    this.qb = this.qb ?? 0.05;
    this.qr = this.qr ?? 0.003;
    this.x.set([h0, 0, 0, 0, 0]);
    this.setDiag([9, 4, 0.02, 4, 0.01]);
  }
  predict(az, dt, extraVar = 0) {
    const x = this.x;
    const a = az - x[2];
    x[0] += x[1] * dt + 0.5 * a * dt * dt;
    x[1] += a * dt;
    x[3] += x[4] * dt;
    const h2 = 0.5 * dt * dt;
    const F = new Float64Array(25);
    for (let i = 0; i < 5; i++) F[i * 5 + i] = 1;
    F[1] = dt; F[2] = -h2; F[7] = -dt; F[19] = dt;
    const sa2 = 0.2 * 0.2 + extraVar;
    const Q = new Float64Array(25);
    Q[0] = h2 * h2 * sa2;
    Q[1] = Q[5] = h2 * dt * sa2;
    Q[6] = dt * dt * sa2;
    Q[12] = 0.003 * 0.003 * dt;
    Q[18] = this.qb * this.qb * dt;
    Q[24] = this.qr * this.qr * dt;
    this.propagate(F, Q);
  }
  baro(z) {
    return this.update(z, this.x[0] + this.x[3], [1, 0, 0, 1, 0], (NOISE.baro * 1.3) ** 2);
  }
  gnssAlt(z) {
    return this.update(z, this.x[0], [1, 0, 0, 0, 0], NOISE.gnssH ** 2);
  }
  get vz() {
    return this.x[1];
  }
  get varVz() {
    return this.P[6];
  }
}

/** Energy-rate Kalman filter: state [E, dE/dt] with E = h + V^2/2g, measured directly; no separate speed-rate estimate. */
export class EnergyKF extends KF {
  constructor() {
    super(2);
    this.reset(0);
  }
  reset(e0) {
    this.qe = this.qe ?? 0.6;
    this.x.set([e0, 0]);
    this.setDiag([9, 1]);
  }
  step(zE, R, dt) {
    const x = this.x;
    x[0] += x[1] * dt;
    const q = this.qe * this.qe;
    this.propagate([1, dt, 0, 1], [(q * dt ** 3) / 3, (q * dt * dt) / 2, (q * dt * dt) / 2, q * dt]);
    this.update(zE, x[0], [1, 0], R);
  }
}

/** Airspeed and its rate from the pitot stream (constant-acceleration model). */
export class AirspeedKF extends KF {
  constructor() {
    super(2);
    this.reset(AC.Vcruise);
  }
  reset(V0) {
    this.x.set([V0, 0]);
    this.setDiag([1, 1]);
  }
  step(z, dt) {
    const x = this.x;
    x[0] += x[1] * dt;
    const qj = (this.qj ?? 0.25) ** 2; // jerk spectral density
    const Q = [(qj * dt ** 3) / 3, (qj * dt * dt) / 2, (qj * dt * dt) / 2, qj * dt];
    this.propagate([1, dt, 0, 1], Q);
    this.update(z, x[0], [1, 0], NOISE.pitot ** 2);
  }
}

/** Horizontal navigation: dead reckoning on airspeed + heading + wind estimate, corrected by GNSS. */
export class Nav {
  constructor() {
    this.reset(0, 0, 0);
  }
  reset(x, y, chi) {
    Object.assign(this, { x, y, chi, phi: 0, V: AC.Vcruise, wu: 0, wv: 0, lastFix: -1, fixes: 0 });
  }
  predict(dt) {
    this.x += (this.V * Math.cos(this.chi) + this.wu) * dt;
    this.y += (this.V * Math.sin(this.chi) + this.wv) * dt;
  }
  gnss(m, t) {
    // timestamp-aware: carry the late fix forward to "now" before blending
    const lat = this.compensate === false ? 0 : t - m.tAcq;
    const vx = this.V * Math.cos(this.chi) + this.wu;
    const vy = this.V * Math.sin(this.chi) + this.wv;
    const kp = this.fixes < 3 ? 1 : 0.3;
    this.x += kp * (m.x + vx * lat - this.x);
    this.y += kp * (m.y + vy * lat - this.y);
    // wind = ground velocity − air velocity, low-passed (~12 s)
    const kw = this.fixes < 10 ? 0.15 : 0.017;
    this.wu += kw * (m.vx - this.V * Math.cos(this.chi) - this.wu);
    this.wv += kw * (m.vy - this.V * Math.sin(this.chi) - this.wv);
    this.lastFix = t;
    this.fixes++;
  }
}

/** The three vertical-air-velocity pipelines compared in the GUI (blueprint Exp. 1–2). */
export class VarioChain {
  /** Optional overrides: {qb, qj, gnss, comp}. Called by the simulator; defaults reproduce the earlier behaviour. */
  configure(o = {}) {
    this.vkf.qb = o.qb ?? 0.05;
    this.vkf5.qb = o.qb ?? 0.05;
    this.vkf5.qr = o.qr ?? 0.003;
    this.enkf.qe = o.qe ?? 0.6;
    this.tiltQ = o.tiltQ === true;
    this.akf.qj = o.qj ?? 0.25;
    this.useGnssAlt = o.gnss !== false;
    this.nav.compensate = o.comp !== false;
  }
  constructor() {
    this.vkf = new VerticalKF();
    this.vkf5 = new VerticalKF5();
    this.enkf = new EnergyKF();
    this.akf = new AirspeedKF();
    this.nav = new Nav();
    this.setFilter('butter', 0.4);
  }
  setFilter(type, fc) {
    this.filterType = type;
    this.fc = fc;
    this.lp = makeFilter(type, fc, RATES.baro);
    this.lp2 = makeFilter(type, fc, RATES.baro); // ablation channel without total-energy compensation
    this.lp3 = makeFilter(type, fc, RATES.baro); // speed-term channel of the complementary pipeline
    this.delay = this.lp.delay;
  }
  reset(g) {
    this.vkf.reset(g.h);
    this.vkf5.reset(g.h);
    this.enkf.reset(g.h + (g.V * g.V) / (2 * G));
    this.vc = 0;
    this.axLp = 0;
    this.vtPrev = null;
    this.wEn = this.wEnA = this.wComp = this.wKf5 = 0;
    this.akf.reset(g.V);
    this.nav.reset(g.x, g.y, g.chi);
    this.setFilter(this.filterType, this.fc);
    this.zPrev = null;
    this.ehPrev = null;
    this.Vraw = g.V;
    this.Vlp = g.V;
    this.wRaw = this.wDsp = this.wKf = 0;
    this.sigKf = 1;
  }
  onImu(m, dt) {
    const ex = this.tiltQ ? (G * Math.tan(this.nav.phi) * 0.0175) ** 2 : 0; // attitude-error allowance in turns (1 deg)
    this.vkf.predict(m.az, dt, ex);
    this.vkf5.predict(m.az, dt, ex);
    this.vc += m.az * dt; // complementary climb rate: integrate the acceleration, the barometer corrects it in onBaro
    if (m.ax !== undefined) this.axLp += (1 - Math.exp(-dt / 0.8)) * (m.ax - this.axLp);
  }
  onAhrs(m) {
    this.nav.phi = m.phi;
    this.nav.chi = m.chi;
  }
  onPitot(m) {
    this.Vraw = m.V;
    this.Vlp += (RATES.pitot === 25 ? 0.22 : 1 - Math.exp((-2 * Math.PI) / RATES.pitot)) * (m.V - this.Vlp); // ~1 Hz one-pole
    this.akf.step(m.V, 1 / RATES.pitot);
    this.nav.V = this.akf.x[0];
  }
  onGnss(m, t) {
    if (this.useGnssAlt !== false) { this.vkf.gnssAlt(m.h); this.vkf5.gnssAlt(m.h); }
    this.nav.gnss(m, t);
  }
  /**
   * Called at the barometer rate. Uses the exact point-mass energy relation
   *   w = ḣ + V·V̇/g + (D − T)·V/(m·g)
   * with each term replaced by what that pipeline can actually measure.
   */
  onBaro(m, motor, polarErr) {
    const dt = 1 / RATES.baro;
    const phi = this.nav.phi;
    const k = 1 + polarErr;
    const thrust = motor ? (AC.motorEta * AC.motorP) / (AC.m * G) : 0;
    this.vkf.baro(m.h);
    this.vkf5.baro(m.h);

    // A — naive: differentiate barometric altitude, add the sink polar. No energy compensation, no filter.
    const hd = this.zPrev === null ? 0 : (m.h - this.zPrev) / dt;
    this.zPrev = m.h;
    this.wRaw = hd + dragSink(Math.max(6, this.Vraw), phi) * k - thrust;

    // B — DSP: total-energy height, differentiated then low-passed with the selected causal filter.
    const eh = m.h + (this.Vlp * this.Vlp) / (2 * G);
    const ed = this.ehPrev === null ? 0 : (eh - this.ehPrev) / dt;
    this.dbg = { eh, ed, sink: dragSink(Math.max(6, this.Vlp), phi) * k - thrust };
    this.ehPrev = eh;
    this.wDsp = this.lp.step(ed) + dragSink(Math.max(6, this.Vlp), phi) * k - thrust;
    this.wNoTE = this.lp2.step(hd) + dragSink(Math.max(6, this.Vlp), phi) * k - thrust;

    // C — Kalman: baro + IMU + GNSS vertical channel, energy term from the airspeed filter.
    const V = Math.max(6, this.akf.x[0]);
    const a = this.akf.x[1];
    this.wKf = this.vkf.vz + (V * a) / G + dragSink(V, phi) * k - thrust;
    const varW = this.vkf.varVz + (V / G) ** 2 * this.akf.P[3] + 0.08 * 0.08;
    this.sigKf = Math.sqrt(Math.max(1e-4, varW));

    // C5 — as C, with a barometer drift-rate state
    this.wKf5 = this.vkf5.vz + (V * a) / G + dragSink(V, phi) * k - thrust;

    // D — energy-rate Kalman filter on E = h + V^2/2g; the polar supplies the drag term
    const Vs = Math.max(6, this.Vlp);
    const Rn = (NOISE.baro * 1.3) ** 2 + ((Vs / G) * 0.09) ** 2;
    this.enkf.step(eh, Rn, dt);
    this.wEn = this.enkf.x[1] + dragSink(Vs, phi) * k - thrust;
    // D2 — same, but the drag term comes from the longitudinal accelerometer: w = dE/dt - V fx / g (no polar, no thrust model)
    this.wEnA = this.enkf.x[1] - (Vs * this.axLp) / G;

    // E — complementary climb-rate filter (acceleration + differenced barometer) plus the low-passed total-energy term
    this.vc += (1 - Math.exp(-2 * Math.PI * this.fc * dt)) * (hd - this.vc);
    const vTerm = (Vs * Vs) / (2 * G);
    const dV = this.vtPrev === null ? 0 : (vTerm - this.vtPrev) / dt;
    this.vtPrev = vTerm;
    this.wComp = this.vc + this.lp3.step(dV) + dragSink(Vs, phi) * k - thrust;
  }
}

/**
 * One-sided CUSUM change detector on vertical air velocity (blueprint §8.2, baseline 3).
 * The reference level and alarm threshold scale with a running estimate of gust intensity,
 * so rough air needs more accumulated evidence than smooth air.
 */
export class Cusum {
  constructor(k = 0.5, h = 1.5) {
    this.k0 = k; // m/s of lift treated as "nothing special" in smooth air
    this.h0 = h; // m of accumulated excess lift needed to alarm in smooth air
    this.reset();
    this.base = 0;
    this.gust = 0.2;
    // gust-adaptation constants (defaults of the paper; exposed for the sensitivity study)
    this.ak = 0.7; this.ah = 1.2; this.ag = 1.25; this.tb = 6; this.tg = 25;
  }
  reset() {
    this.g = 0;
  }
  get k() {
    return this.k0 + this.ak * this.gust;
  }
  get h() {
    return this.h0 * (1 + this.ah * this.gust);
  }
  /** armed=false still tracks the gust level but cannot alarm. */
  step(w, dt, armed = true) {
    this.base += ((w - this.base) * dt) / this.tb;
    this.gust += ((this.ag * Math.abs(w - this.base) - this.gust) * dt) / this.tg;
    if (!armed) {
      this.g = 0;
      return false;
    }
    this.g = Math.max(0, this.g + (w - this.k) * dt);
    return this.g > this.h;
  }
}

const LIM = { wMin: 0.2, wMax: 10, sMin: 15, sMax: 160 };

/** Particle filter over θ = [xc, yc, wmax, σ]; drift is supplied by the wind estimate. */
export class ThermalPF {
  constructor(rng, N = 400, q = {}) {
    this.rng = rng;
    this.N = N;
    this.kind = 'pf';
    // random-walk process noise per √s: centre [m], strength [m/s], spread [m]
    this.q = { xy: 3, w: 0.12, s: 1.5, ...q };
    this.px = new Float64Array(N);
    this.py = new Float64Array(N);
    this.pw = new Float64Array(N);
    this.ps = new Float64Array(N);
    this.wt = new Float64Array(N);
    this.est = null;
  }
  init(x, y, w0) {
    const r = this.rng;
    for (let i = 0; i < this.N; i++) {
      this.px[i] = x + r.gauss(75);
      this.py[i] = y + r.gauss(75);
      this.pw[i] = r.uni(Math.max(0.6, w0 * 0.8), Math.max(4.5, w0 * 3));
      this.ps[i] = r.uni(25, 115);
      this.wt[i] = 1 / this.N;
    }
    this.summarise();
  }
  predict(dt, u, v) {
    const r = this.rng;
    const sq = Math.sqrt(dt);
    for (let i = 0; i < this.N; i++) {
      this.px[i] += u * dt + r.gauss(this.q.xy * sq);
      this.py[i] += v * dt + r.gauss(this.q.xy * sq);
      this.pw[i] = clamp(this.pw[i] + r.gauss(this.q.w * sq), LIM.wMin, LIM.wMax);
      this.ps[i] = clamp(this.ps[i] + r.gauss(this.q.s * sq), LIM.sMin, LIM.sMax);
    }
  }
  update(x, y, z, R) {
    const N = this.N;
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const dx = x - this.px[i];
      const dy = y - this.py[i];
      const pred = this.pw[i] * Math.exp(-(dx * dx + dy * dy) / (2 * this.ps[i] * this.ps[i]));
      const e = z - pred;
      this.wt[i] *= Math.exp((-0.5 * e * e) / R) + 1e-300;
      sum += this.wt[i];
    }
    let neff = 0;
    for (let i = 0; i < N; i++) {
      this.wt[i] /= sum;
      neff += this.wt[i] * this.wt[i];
    }
    if (1 / neff < N * 0.5) this.resample();
    this.summarise();
  }
  resample() {
    const N = this.N;
    const r = this.rng;
    const nx = new Float64Array(N);
    const ny = new Float64Array(N);
    const nw = new Float64Array(N);
    const nS = new Float64Array(N);
    let c = this.wt[0];
    let i = 0;
    const u0 = r.next() / N;
    for (let j = 0; j < N; j++) {
      const u = u0 + j / N;
      while (u > c && i < N - 1) c += this.wt[++i];
      nx[j] = this.px[i] + r.gauss(2.5); // roughening keeps diversity after resampling
      ny[j] = this.py[i] + r.gauss(2.5);
      nw[j] = clamp(this.pw[i] + r.gauss(0.05), LIM.wMin, LIM.wMax);
      nS[j] = clamp(this.ps[i] + r.gauss(1.2), LIM.sMin, LIM.sMax);
    }
    this.px = nx;
    this.py = ny;
    this.pw = nw;
    this.ps = nS;
    this.wt.fill(1 / N);
  }
  summarise() {
    const N = this.N;
    let mx = 0, my = 0, mw = 0, ms = 0;
    for (let i = 0; i < N; i++) {
      const q = this.wt[i];
      mx += q * this.px[i];
      my += q * this.py[i];
      mw += q * this.pw[i];
      ms += q * this.ps[i];
    }
    let Pxx = 0, Pxy = 0, Pyy = 0, vw = 0, vs = 0;
    for (let i = 0; i < N; i++) {
      const q = this.wt[i];
      const dx = this.px[i] - mx;
      const dy = this.py[i] - my;
      Pxx += q * dx * dx;
      Pxy += q * dx * dy;
      Pyy += q * dy * dy;
      vw += q * (this.pw[i] - mw) ** 2;
      vs += q * (this.ps[i] - ms) ** 2;
    }
    this.est = { x: mx, y: my, w: mw, s: ms, Pxx, Pxy, Pyy, sdW: Math.sqrt(vw), sdS: Math.sqrt(vs), posSd: Math.sqrt(0.5 * (Pxx + Pyy)) };
  }
  get() {
    return this.est;
  }
}

/** Extended Kalman filter over the same four parameters (classical recursive baseline). */
export class ThermalEKF extends KF {
  constructor() {
    super(4);
    this.kind = 'ekf';
    this.est = null;
  }
  init(x, y, w0) {
    this.x.set([x, y, clamp(w0 * 1.6, 1.5, 6), 60]);
    this.setDiag([70 * 70, 70 * 70, 4, 30 * 30]);
    this.summarise();
  }
  predict(dt, u, v) {
    this.x[0] += u * dt;
    this.x[1] += v * dt;
    const P = this.P;
    // same random-walk process noise as the particle filter, for a like-for-like comparison
    P[0] += 9 * dt;
    P[5] += 9 * dt;
    P[10] += 0.0144 * dt;
    P[15] += 2.25 * dt;
  }
  update(x, y, z, R) {
    const [xc, yc, wm, s] = this.x;
    const dx = x - xc;
    const dy = y - yc;
    const r2 = dx * dx + dy * dy;
    const e = Math.exp(-r2 / (2 * s * s));
    const hx = wm * e;
    const H = [(hx * dx) / (s * s), (hx * dy) / (s * s), e, (hx * r2) / (s * s * s)];
    super.update(z, hx, H, R);
    this.x[2] = clamp(this.x[2], LIM.wMin, LIM.wMax);
    this.x[3] = clamp(this.x[3], LIM.sMin, LIM.sMax);
    this.summarise();
  }
  summarise() {
    const P = this.P;
    const [x, y, w, s] = this.x;
    const Pxx = Math.max(0, P[0]);
    const Pyy = Math.max(0, P[5]);
    this.est = { x, y, w, s, Pxx, Pxy: P[1], Pyy, sdW: Math.sqrt(Math.max(0, P[10])), sdS: Math.sqrt(Math.max(0, P[15])), posSd: Math.sqrt(0.5 * (Pxx + Pyy)) };
  }
  get() {
    return this.est;
  }
}
