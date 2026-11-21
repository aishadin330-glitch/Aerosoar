// Causal DSP blocks (blueprint §7): low-pass filters with known delay, and a Welch PSD.
// Everything here uses only past samples, so it could run onboard.

class Butter2 {
  constructor(fc, fs) {
    const K = Math.tan((Math.PI * fc) / fs);
    const n = 1 / (1 + Math.SQRT2 * K + K * K);
    this.b0 = K * K * n;
    this.b1 = 2 * this.b0;
    this.b2 = this.b0;
    this.a1 = 2 * (K * K - 1) * n;
    this.a2 = (1 - Math.SQRT2 * K + K * K) * n;
    this.z1 = 0;
    this.z2 = 0;
    this.primed = false;
    this.delay = Math.SQRT2 / (2 * Math.PI * fc); // low-frequency group delay [s]
  }
  step(x) {
    if (!this.primed) {
      // start in steady state to avoid a start-up transient
      this.z1 = x * (1 - this.b0);
      this.z2 = x * (this.b2 - this.a2);
      this.primed = true;
    }
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }
}

class MovAvg {
  constructor(fc, fs) {
    this.N = Math.max(1, Math.round((0.443 * fs) / fc)); // −3 dB point near fc
    this.buf = new Float32Array(this.N);
    this.i = 0;
    this.sum = 0;
    this.count = 0;
    this.delay = (this.N - 1) / (2 * fs);
  }
  step(x) {
    this.sum += x - this.buf[this.i];
    this.buf[this.i] = x;
    this.i = (this.i + 1) % this.N;
    if (this.count < this.N) this.count++;
    return this.sum / this.count;
  }
}

class Ema {
  constructor(fc, fs) {
    this.a = 1 - Math.exp((-2 * Math.PI * fc) / fs);
    this.y = null;
    this.delay = 1 / (2 * Math.PI * fc);
  }
  step(x) {
    this.y = this.y === null ? x : this.y + this.a * (x - this.y);
    return this.y;
  }
}

export const FILTERS = { butter: 'Butterworth 2', movavg: 'Moving average', ema: 'One-pole' };

export function makeFilter(type, fc, fs) {
  if (type === 'movavg') return new MovAvg(fc, fs);
  if (type === 'ema') return new Ema(fc, fs);
  return new Butter2(fc, fs);
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/**
 * Welch power spectral density: Hann window, 50 % overlap, one-sided, units²/Hz.
 * x: array of samples (oldest first). Returns null until one full segment exists.
 */
export function welch(x, fs, nseg = 256) {
  if (x.length < nseg) return null;
  const win = new Float64Array(nseg);
  let U = 0;
  for (let i = 0; i < nseg; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (nseg - 1));
    U += win[i] * win[i];
  }
  const half = nseg / 2;
  const p = new Float64Array(half + 1);
  let segs = 0;
  const re = new Float64Array(nseg);
  const im = new Float64Array(nseg);
  for (let s = 0; s + nseg <= x.length; s += half) {
    let mean = 0;
    for (let i = 0; i < nseg; i++) mean += x[s + i];
    mean /= nseg;
    for (let i = 0; i < nseg; i++) {
      re[i] = (x[s + i] - mean) * win[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= half; k++) {
      const m = (re[k] * re[k] + im[k] * im[k]) / (fs * U);
      p[k] += k === 0 || k === half ? m : 2 * m;
    }
    segs++;
  }
  const f = new Float64Array(half + 1);
  for (let k = 0; k <= half; k++) {
    p[k] /= segs;
    f[k] = (k * fs) / nseg;
  }
  return { f, p, segs };
}
