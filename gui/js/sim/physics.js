// Truth models: 3-DOF point-mass glider and the synthetic air mass.
// Frame: local East-North-Up. x east [m], y north [m], h up [m].
// chi = air-relative heading, measured from +x toward +y [rad]. gam > 0 climbs. phi > 0 turns toward +chi.
// w > 0 is rising air. All of this is simulator truth and is never handed to an estimator.
import { G, RHO, D2R, clamp } from './util.js';

// Blueprint §4.2 starting values (assumptions, not a measured airframe).
export const AC = {
  m: 2.0,
  b: 2.0,
  S: 0.45,
  CD0: 0.018,
  e: 0.85,
  CLmax: 1.25,
  phiMax: 45 * D2R,
  rollRate: 70 * D2R,
  tauPhi: 0.35,
  Vcruise: 14,
  Vthermal: 10.5,
  motorP: 150, // electrical W
  motorEta: 0.6,
};
AC.AR = (AC.b * AC.b) / AC.S;
AC.k = 1 / (Math.PI * AC.e * AC.AR);

/**
 * Still-air sink D·V/(m·g) [m/s, positive down] for airspeed V and bank phi,
 * assuming load factor 1/cos(phi). cd0Scale lets an estimator carry a wrong polar.
 */
export function dragSink(V, phi, cd0Scale = 1) {
  const q = 0.5 * RHO * V * V;
  const n = 1 / Math.max(0.3, Math.cos(phi));
  const CL = clamp((n * AC.m * G) / (q * AC.S), 0, AC.CLmax * 1.3);
  const CD = AC.CD0 * cd0Scale + AC.k * CL * CL;
  return (q * AC.S * CD * V) / (AC.m * G);
}

export class Glider {
  constructor() {
    this.reset();
  }
  reset(x = 0, y = 0, h = 300, chi = 0) {
    Object.assign(this, { x, y, h, chi, V: AC.Vcruise, gam: -0.06, phi: 0 });
    this.phiCmd = 0;
    this.Vcmd = AC.Vcruise;
    this.motor = false;
    this.hdot = this.V * Math.sin(this.gam);
    this.az = 0; // vertical inertial acceleration [m/s²]
    this.vx = 0;
    this.vy = 0;
    this.CL = 0.4;
  }
  step(dt, wind) {
    const { m, S } = AC;
    // roll: first-order response with a rate limit
    const pc = clamp(this.phiCmd, -AC.phiMax, AC.phiMax);
    const dphi = clamp(((pc - this.phi) / AC.tauPhi) * dt, -AC.rollRate * dt, AC.rollRate * dt);
    this.phi += dphi;

    const V = this.V;
    const q = 0.5 * RHO * V * V;
    const cphi = Math.cos(this.phi);
    const T = this.motor ? (AC.motorEta * AC.motorP) / Math.max(V, 6) : 0;

    // pitch autopilot: hold commanded airspeed by steering the flight-path angle
    const qc = 0.5 * RHO * this.Vcmd * this.Vcmd;
    const CLt = clamp((m * G) / (qc * S * cphi), 0.05, AC.CLmax);
    const Dt = qc * S * (AC.CD0 + AC.k * CLt * CLt);
    const gamEq = Math.asin(clamp((T - Dt) / (m * G), -0.6, 0.6));
    const gamCmd = clamp(gamEq + 0.06 * (V - this.Vcmd), -0.55, 0.55);
    const gamDotDes = 2.0 * (gamCmd - this.gam);
    const CL = clamp((m * V * gamDotDes + m * G * Math.cos(this.gam)) / (q * S * cphi), -0.2, AC.CLmax);
    const CD = AC.CD0 + AC.k * CL * CL;
    const L = q * S * CL;
    const D = q * S * CD;
    this.CL = CL;

    // air-relative point-mass dynamics (wind-gradient terms neglected, see RESEARCH_PLAN §4)
    const Vdot = (T - D) / m - G * Math.sin(this.gam);
    const gamDot = (L * cphi - m * G * Math.cos(this.gam)) / (m * V);
    const chiDot = (L * Math.sin(this.phi)) / (m * V * Math.cos(this.gam));
    this.V = Math.max(5, V + Vdot * dt);
    this.gam = clamp(this.gam + gamDot * dt, -1.0, 1.0);
    this.chi += chiDot * dt;

    // ground-relative kinematics: wind is added exactly once, here
    const cg = Math.cos(this.gam);
    this.vx = this.V * cg * Math.cos(this.chi) + wind.u;
    this.vy = this.V * cg * Math.sin(this.chi) + wind.v;
    const hdot = this.V * Math.sin(this.gam) + wind.w;
    this.az = (hdot - this.hdot) / dt;
    this.hdot = hdot;
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.h = Math.max(0, this.h + hdot * dt);
  }
}

let thermalId = 0;

export class Atmosphere {
  constructor(rng) {
    this.rng = rng;
    this.zi = 620; // convective-layer top [m]
    this.reset();
  }
  reset() {
    this.thermals = [];
    this.u = 0;
    this.v = 0;
    this.turb = 0.3;
    this.sinkRing = false;
    this.driftFactor = 0.9;
    this.g1 = 0;
    this.g2 = 0;
    thermalId = 0;
  }
  /** Meteorological convention: direction the wind blows FROM, degrees clockwise from north. */
  setWind(speed, fromDeg) {
    const a = fromDeg * D2R;
    this.u = -speed * Math.sin(a);
    this.v = -speed * Math.cos(a);
  }
  addThermal(o) {
    const th = {
      id: ++thermalId,
      x: o.x,
      y: o.y,
      wmax: o.wmax,
      sigma: o.sigma,
      life: o.life || 'steady',
      period: o.period || this.rng.uni(140, 260),
      phase: this.rng.uni(0, 6.28),
      age: 0,
      s: 1,
    };
    this.thermals.push(th);
    return th;
  }
  step(dt) {
    const du = this.u * this.driftFactor * dt;
    const dv = this.v * this.driftFactor * dt;
    for (const th of this.thermals) {
      th.x += du;
      th.y += dv;
      th.age += dt;
      th.s = th.life === 'pulse' ? 0.7 + 0.3 * Math.sin((2 * Math.PI * th.age) / th.period + th.phase) : 1;
    }
    // vertical gust: Ornstein–Uhlenbeck process passed through a short lag (keeps it differentiable)
    const tau = 3.5;
    this.g1 += (-this.g1 / tau) * dt + this.turb * Math.sqrt((2 * dt) / tau) * this.rng.gauss();
    this.g2 += ((this.g1 - this.g2) * dt) / 0.45;
  }
  /** Vertical profile: lift fades in above the ground and dies under the layer top. */
  hprof(h) {
    return clamp(h / 80, 0, 1) * clamp((this.zi - h) / 160, 0, 1);
  }
  shape(r2, sigma) {
    const a = r2 / (2 * sigma * sigma);
    return this.sinkRing ? Math.exp(-a) * (1 - a) : Math.exp(-a);
  }
  /** Thermal part of the vertical wind only. */
  thermalW(x, y, h) {
    let w = 0;
    const hp = this.hprof(h);
    for (const th of this.thermals) {
      const dx = x - th.x;
      const dy = y - th.y;
      w += th.wmax * th.s * hp * this.shape(dx * dx + dy * dy, th.sigma);
    }
    return w;
  }
  sample(x, y, h) {
    const wT = this.thermalW(x, y, h);
    return { u: this.u, v: this.v, w: wT + this.g2, wT };
  }
  /** Thermal with the largest contribution at (x,y); used for scoring and the oracle only. */
  dominant(x, y, maxDist = 260) {
    let best = null;
    let bw = -1;
    for (const th of this.thermals) {
      const d2 = (x - th.x) ** 2 + (y - th.y) ** 2;
      if (d2 > maxDist * maxDist) continue;
      const c = th.wmax * th.s * Math.exp(-d2 / (2 * th.sigma * th.sigma));
      if (c > bw) {
        bw = c;
        best = th;
      }
    }
    return best;
  }
}
