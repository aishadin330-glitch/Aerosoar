// Imperfect onboard sensors (blueprint §6). Each stream has its own rate, noise, bias and faults,
// and every sample carries the time it was acquired so nothing is aligned by array index.
import { D2R } from './util.js';

export const RATES = { imu: 100, ahrs: 50, pitot: 25, baro: 20, gnss: 5 }; // Hz (simulation assumptions)
export const NOISE = {
  baro: 0.12, // m
  gnssXY: 1.2, // m
  gnssH: 2.0, // m
  gnssV: 0.12, // m/s
  pitot: 0.25, // m/s
  phi: 1.0 * D2R,
  chi: 2.0 * D2R,
  accel: 0.12, // m/s²
};
export const GNSS_LATENCY = 0.15; // s

export class Sensors {
  constructor(rng) {
    this.rng = rng;
    this.reset();
  }
  reset() {
    this.n = 0;
    this.baroBias = this.rng.gauss(0.8);
    this.accBias = this.rng.gauss(0.04);
    this.pitotLag = null;
    this.queue = []; // GNSS fixes waiting out their latency
    this.gnssOk = true;
    this.outageUntil = 0;
    this.nextOutage = 12;
  }
  /**
   * Advance one 100 Hz tick. Returns the samples that arrive on this tick
   * ({imu, ahrs?, pitot?, baro?, gnss?}); absent keys mean "no new sample".
   */
  step(t, dt, g, cfg) {
    const r = this.rng;
    const ns = cfg.noise;
    const out = {};
    const k = this.n++;

    // IMU: gravity-compensated vertical acceleration
    this.accBias += r.gauss(0.002) * Math.sqrt(dt);
    const vib = cfg.stress?.vibAmp ? cfg.stress.vibAmp * Math.sin(2 * Math.PI * (cfg.stress.vibF || 30) * t) : 0;
    if (k % Math.round(100 / RATES.imu) === 0) out.imu = { t, az: g.az + this.accBias + vib + r.gauss(NOISE.accel * ns) };

    if (k % 2 === 0) {
      out.ahrs = { t, phi: g.phi + r.gauss(NOISE.phi * ns), chi: g.chi + r.gauss(NOISE.chi * ns) };
    }

    // pitot: scale error, pneumatic lag, noise
    const scale = 1.01 + (cfg.faults.pitot ? 0.07 : 0);
    if (this.pitotLag === null) this.pitotLag = g.V * scale;
    this.pitotLag += ((g.V * scale - this.pitotLag) * dt) / (cfg.pitotTau ?? 0.12);
    if (k % Math.round(100 / RATES.pitot) === 0) out.pitot = { t, V: this.pitotLag + r.gauss(NOISE.pitot * ns) };

    // barometer: slow random-walk bias, optional steady pressure drift
    this.baroBias += r.gauss(0.02) * Math.sqrt(dt) + (cfg.faults.baro ? 0.08 * dt : 0);
    if (k % Math.round(100 / RATES.baro) === 0) {
      let nb = r.gauss(NOISE.baro * ns);
      if (cfg.stress?.tauBaro > 0) {
        const a = Math.exp(-0.05 / cfg.stress.tauBaro);
        this.baroN = a * (this.baroN || 0) + Math.sqrt(1 - a * a) * nb;
        nb = this.baroN;
      }
      out.baro = { t, h: g.h + this.baroBias + nb };
    }

    // GNSS: 5 Hz, delivered late, optional outages
    if (cfg.faults.gnss) {
      if (t >= this.nextOutage) {
        this.outageUntil = t + r.uni(6, 12);
        this.nextOutage = this.outageUntil + r.uni(14, 26);
      }
    } else {
      this.outageUntil = 0;
      this.nextOutage = t + 6;
    }
    this.gnssOk = t >= this.outageUntil;
    if (k % 20 === 0 && this.gnssOk) {
      this.queue.push({
        tAcq: t,
        tArr: t + (cfg.gnssDelay ?? GNSS_LATENCY),
        x: g.x + r.gauss(NOISE.gnssXY * ns),
        y: g.y + r.gauss(NOISE.gnssXY * ns),
        h: g.h + r.gauss(NOISE.gnssH * ns),
        vx: g.vx + r.gauss(NOISE.gnssV * ns),
        vy: g.vy + r.gauss(NOISE.gnssV * ns),
      });
    }
    if (this.queue.length && this.queue[0].tArr <= t) out.gnss = this.queue.shift();
    return out;
  }
}
