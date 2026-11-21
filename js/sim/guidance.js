// Mission guidance (blueprint §10): fly directly to a distant destination and circle only
// inside a thermal to regain altitude, then resume the direct course. An interpretable
// finite-state machine with a conventional thermal-centering law. It receives estimates only
// (navigation, vertical air velocity, thermal estimate with uncertainty), never simulator truth.
import { G, D2R, clamp, wrapPi } from './util.js';
import { AC, dragSink } from './physics.js';

export const STATES = ['CRUISE', 'ASSESS', 'CLIMB', 'MOTOR', 'ARRIVED'];

export const LIMITS = {
  ceiling: 420, // m: leave the thermal at this altitude
  reentry: 330, // m: after reaching the ceiling, thermals are ignored until below this altitude
  hLow: 80, // m: engage the motor
  hRelaunch: 290, // m: disengage the motor
  hArrive: 50, // m: minimum arrival altitude used for the final-glide calculation
  arriveRadius: 60, // m: destination capture radius
  glideMargin: 1.25, // safety factor on the altitude required for the final glide
};

export class Guidance {
  constructor(rng, goal) {
    this.rng = rng;
    this.goal = goal;
    this.reset();
  }
  reset() {
    this.state = 'CRUISE';
    this.reason = 'Proceeding directly to destination';
    this.tState = 0;
    this.dir = 1;
    this.climbAvg = 0;
    this.weakT = 0;
    this.inhibit = 3;
    this.topOut = false;
    this.wSum = 0;
    this.wN = 0;
    this.orbitR = 25;
    this.dist = Math.hypot(this.goal.x, this.goal.y);
    this.need = 0;
    this.glideSlope = 0.08;
    this.finalGlide = false;
    this.cmd = { phi: 0, V: AC.Vcruise, motor: false };
  }
  go(state, reason, I) {
    const from = this.state;
    this.state = state;
    this.reason = reason;
    this.tState = 0;
    I.emit('state', { from, to: state, reason });
  }
  steerTo(I, tx, ty, maxBank) {
    const want = Math.atan2(ty - I.y, tx - I.x);
    return clamp(1.3 * wrapPi(want - I.chi), -maxBank, maxBank);
  }
  /** Vector-field orbit about the estimated (drifting) centre, flown in the air-mass frame. */
  orbit(I, cx, cy, R) {
    const dx = I.x - cx;
    const dy = I.y - cy;
    const d = Math.hypot(dx, dy);
    const th = Math.atan2(dy, dx);
    const want = th + this.dir * (Math.PI / 2 + Math.atan((1.6 * (d - R)) / R));
    const ff = this.dir * Math.atan((I.V * I.V) / (G * R)) * Math.exp(-(((d - R) / R) ** 2));
    return clamp(ff + 1.4 * wrapPi(want - I.chi), -AC.phiMax, AC.phiMax);
  }
  /** Leave the thermal (or abandon a candidate) and resume the direct course. */
  resume(I, reason) {
    I.stopTrack();
    this.inhibit = 7;
    this.go('CRUISE', reason, I);
  }
  update(dt, I) {
    const c = this.cmd;
    const e = I.est;
    const model = I.kind !== 'reactive';
    const goal = this.goal;
    this.tState += dt;
    this.inhibit -= dt;
    if (I.h > LIMITS.ceiling - 20) this.topOut = true;
    if (I.h < LIMITS.reentry) this.topOut = false;

    // final-glide calculation: altitude required to reach the destination at cruise speed,
    // using the estimated wind component along the remaining track
    const gx = goal.x - I.x;
    const gy = goal.y - I.y;
    this.dist = Math.hypot(gx, gy);
    const along = (I.wu * gx + I.wv * gy) / Math.max(1, this.dist);
    this.glideSlope = (LIMITS.glideMargin * dragSink(AC.Vcruise, 0)) / Math.max(4, AC.Vcruise + along);
    this.need = LIMITS.hArrive + this.glideSlope * this.dist;
    this.finalGlide = I.h > this.need;

    if (this.state === 'ARRIVED') {
      c.phi = 0;
      c.motor = false;
      return c;
    }
    if (this.dist < LIMITS.arriveRadius) {
      I.stopTrack();
      c.phi = 0;
      c.motor = false;
      this.go('ARRIVED', 'Destination reached', I);
      return c;
    }
    if (this.state !== 'MOTOR' && I.h < LIMITS.hLow) {
      I.stopTrack();
      this.go('MOTOR', 'Below minimum altitude; motor engaged, course maintained', I);
    }

    switch (this.state) {
      case 'CRUISE': {
        c.V = AC.Vcruise;
        c.motor = false;
        c.phi = this.steerTo(I, goal.x, goal.y, 30 * D2R);
        if (this.tState > 6) this.reason = this.finalGlide ? 'Final glide to destination' : 'Proceeding directly to destination';
        if (this.inhibit <= 0 && !this.topOut && !this.finalGlide && I.trip) {
          I.startTrack();
          this.dir = this.rng.next() < 0.5 ? 1 : -1;
          this.wSum = 0;
          this.wN = 0;
          this.go('ASSESS', 'Sustained lift detected; assessing thermal', I);
        }
        break;
      }
      case 'ASSESS': {
        c.V = AC.Vthermal;
        c.phi = this.tState < 2 ? 0 : this.dir * 30 * D2R; // continue into the lift, then circle
        if (this.tState > 2) {
          this.wSum += I.w;
          this.wN++;
        }
        const wMean = this.wN ? this.wSum / this.wN : 0;
        if (this.tState > 9) {
          if (!model) {
            if (wMean > 0.45) {
              this.climbAvg = wMean;
              this.weakT = 0;
              this.go('CLIMB', `Mean lift ${wMean.toFixed(1)} m/s; climbing by variometer rule`, I);
            } else if (this.tState > 15) this.resume(I, 'Thermal rejected: lift not sustained; course resumed');
          } else if (e && e.w > 0.8 && e.posSd < 45 && wMean > 0.1) {
            this.orbitR = clamp(0.55 * e.s, 16, 40);
            this.climbAvg = Math.max(0.3, wMean);
            this.weakT = 0;
            this.go('CLIMB', `Thermal accepted (centre uncertainty ±${e.posSd.toFixed(0)} m); climbing`, I);
          } else if (this.tState > 22 || !e) this.resume(I, 'Thermal rejected: insufficient strength or confidence; course resumed');
        }
        break;
      }
      case 'CLIMB': {
        c.V = AC.Vthermal;
        if (model) {
          if (!e) {
            this.resume(I, 'Thermal estimate unavailable; course resumed');
            break;
          }
          this.orbitR += (clamp(0.55 * e.s, 16, 40) - this.orbitR) * dt * 0.5;
          c.phi = this.orbit(I, e.x, e.y, this.orbitR);
        } else {
          // classical variometer rule: reduce bank while lift increases, increase bank while it decreases
          c.phi = this.dir * clamp((32 - 14 * I.wDot) * D2R, 14 * D2R, AC.phiMax);
        }
        this.climbAvg += ((I.vz - this.climbAvg) * dt) / 12;
        this.weakT = this.climbAvg < 0.15 ? this.weakT + dt : 0;
        if (I.h > LIMITS.ceiling) this.resume(I, 'Operating ceiling reached; course resumed');
        else if (I.h > this.need + 40) this.resume(I, 'Final-glide altitude attained; course resumed');
        else if (this.tState > 20 && this.weakT > 10) this.resume(I, 'Climb rate below threshold; course resumed');
        else if (model && e.posSd > 90) this.resume(I, 'Position uncertainty exceeds limit; course resumed');
        break;
      }
      case 'MOTOR': {
        c.motor = true;
        c.V = 12;
        c.phi = this.steerTo(I, goal.x, goal.y, 25 * D2R);
        if (I.h > LIMITS.hRelaunch) {
          c.motor = false;
          this.inhibit = 4;
          this.go('CRUISE', 'Recovery altitude reached; motor disengaged', I);
        }
        break;
      }
    }
    return c;
  }
}
