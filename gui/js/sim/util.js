// Shared constants and small helpers. No DOM access — importable from Node for headless tests.
export const G = 9.81;
export const RHO = 1.225;
export const D2R = Math.PI / 180;
export const R2D = 180 / Math.PI;

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

export function wrapPi(a) {
  a = (a + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

/** Seeded generator (mulberry32) so every run is reproducible from its seed. */
export class RNG {
  constructor(seed = 1) {
    this.s = seed >>> 0;
    this.spare = null;
  }
  next() {
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uni(a = 0, b = 1) {
    return a + (b - a) * this.next();
  }
  gauss(sd = 1) {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v * sd;
    }
    let u, v, s;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * m;
    return u * m * sd;
  }
}

/** Fixed-size ring buffer of named Float32 channels (oldest → newest via at()). */
export class Ring {
  constructor(n, fields) {
    this.n = n;
    this.fields = fields;
    this.len = 0;
    this.head = 0;
    for (const f of fields) this[f] = new Float32Array(n);
  }
  push(o) {
    for (const f of this.fields) this[f][this.head] = o[f];
    this.head = (this.head + 1) % this.n;
    if (this.len < this.n) this.len++;
  }
  /** index of the i-th oldest sample */
  idx(i) {
    return (this.head - this.len + i + this.n) % this.n;
  }
  clear() {
    this.len = 0;
    this.head = 0;
  }
}
