// In-browser inference for the trained networks (weights exported from ml/train.py).
// Supports both architectures that were compared: a 2-layer LSTM and a pre-norm Transformer encoder.
// Input: WIN×NF window (row-major Float32Array). Output: existence probability and thermal parameters
// with calibrated standard deviations, all relative to the current navigation position.

const sig = (x) => 1 / (1 + Math.exp(-x));

/** y = W·x + b for W stored as rows (out × in). */
function linear(W, b, x) {
  const y = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = b[i];
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    y[i] = s;
  }
  return y;
}

function layerNorm(x, g, b) {
  const n = x.length;
  let m = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  const r = 1 / Math.sqrt(v / n + 1e-5);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = (x[i] - m) * r * g[i] + b[i];
  return y;
}

export class MLModel {
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`model not found: ${url}`);
    return new MLModel(await r.json());
  }
  constructor(j) {
    this.name = j.name;
    this.win = j.win;
    this.nf = j.nf;
    this.k = j.var_scale;
    this.w = j.weights;
    this.testX = j.test_x;
    this.testY = j.test_y;
  }
  forwardLSTM(x) {
    const w = this.w;
    const H = w['rnn.weight_hh_l0'][0].length;
    let inp = Array.from({ length: this.win }, (_, t) => x.subarray(t * this.nf, (t + 1) * this.nf));
    for (let l = 0; l < 2; l++) {
      const Wi = w[`rnn.weight_ih_l${l}`], Wh = w[`rnn.weight_hh_l${l}`];
      const bi = w[`rnn.bias_ih_l${l}`], bh = w[`rnn.bias_hh_l${l}`];
      let h = new Float32Array(H), c = new Float32Array(H);
      const outs = [];
      for (const xt of inp) {
        const a = linear(Wi, bi, xt);
        const b = linear(Wh, bh, h);
        const nh = new Float32Array(H);
        for (let k = 0; k < H; k++) {
          const i = sig(a[k] + b[k]), f = sig(a[H + k] + b[H + k]);
          const g = Math.tanh(a[2 * H + k] + b[2 * H + k]), o = sig(a[3 * H + k] + b[3 * H + k]);
          c[k] = f * c[k] + i * g;
          nh[k] = o * Math.tanh(c[k]);
        }
        h = nh;
        outs.push(h);
      }
      inp = outs;
    }
    const z = linear(w['head.0.weight'], w['head.0.bias'], inp[inp.length - 1]).map((v) => Math.max(0, v));
    return linear(w['head.2.weight'], w['head.2.bias'], z);
  }
  forwardTransformer(x) {
    const w = this.w;
    const d = w['inp.weight'].length;
    const heads = 4, dh = d / heads;
    let T = Array.from({ length: this.win }, (_, t) => {
      const e = linear(w['inp.weight'], w['inp.bias'], x.subarray(t * this.nf, (t + 1) * this.nf));
      for (let i = 0; i < d; i++) e[i] += w['pos'][t][i];
      return e;
    });
    for (let l = 0; l < 2; l++) {
      const p = `enc.layers.${l}.`;
      const N = T.map((t) => layerNorm(t, w[p + 'norm1.weight'], w[p + 'norm1.bias']));
      const qkv = N.map((t) => linear(w[p + 'self_attn.in_proj_weight'], w[p + 'self_attn.in_proj_bias'], t));
      const att = T.map(() => new Float32Array(d));
      for (let h = 0; h < heads; h++) {
        for (let i = 0; i < this.win; i++) {
          const sc = new Float32Array(this.win);
          let mx = -1e9;
          for (let j = 0; j < this.win; j++) {
            let s = 0;
            for (let k = 0; k < dh; k++) s += qkv[i][h * dh + k] * qkv[j][d + h * dh + k];
            sc[j] = s / Math.sqrt(dh);
            if (sc[j] > mx) mx = sc[j];
          }
          let sum = 0;
          for (let j = 0; j < this.win; j++) sum += (sc[j] = Math.exp(sc[j] - mx));
          for (let j = 0; j < this.win; j++) for (let k = 0; k < dh; k++) att[i][h * dh + k] += (sc[j] / sum) * qkv[j][2 * d + h * dh + k];
        }
      }
      T = T.map((t, i) => {
        const o = linear(w[p + 'self_attn.out_proj.weight'], w[p + 'self_attn.out_proj.bias'], att[i]);
        for (let k = 0; k < d; k++) o[k] += t[k];
        const f = layerNorm(o, w[p + 'norm2.weight'], w[p + 'norm2.bias']);
        const u = linear(w[p + 'linear1.weight'], w[p + 'linear1.bias'], f).map((v) => Math.max(0, v));
        const y = linear(w[p + 'linear2.weight'], w[p + 'linear2.bias'], u);
        for (let k = 0; k < d; k++) y[k] += o[k];
        return y;
      });
    }
    const last = layerNorm(T[T.length - 1], w['norm.weight'], w['norm.bias']);
    const z = linear(w['head.0.weight'], w['head.0.bias'], last).map((v) => Math.max(0, v));
    return linear(w['head.2.weight'], w['head.2.bias'], z);
  }
  raw(x) {
    return this.name === 'lstm' ? this.forwardLSTM(x) : this.forwardTransformer(x);
  }
  /** Physical units; centre offsets relative to the current navigation position. */
  predict(x) {
    const o = this.raw(x);
    const sd = (i) => Math.sqrt(Math.exp(Math.min(3, Math.max(-6, o[5 + i]))) * this.k);
    return {
      p: sig(o[0]),
      cx: o[1] * 100, cy: o[2] * 100, w: Math.max(0, o[3] * 4), s: Math.max(15, o[4] * 60),
      sdx: sd(0) * 100, sdy: sd(1) * 100, sdw: sd(2) * 4, sds: sd(3) * 60,
    };
  }
  /** Largest deviation from the PyTorch outputs on the exported test windows. */
  selfCheck() {
    let m = 0;
    this.testX.forEach((win, i) => {
      const o = this.raw(Float32Array.from(win.flat()));
      o.forEach((v, k) => (m = Math.max(m, Math.abs(v - this.testY[i][k]))));
    });
    return m;
  }
}
