// Two-dimensional displays: vertical-air-velocity time series and flight profile.
import { STATES, LIMITS, ROUTE } from '../sim/sim.js';
import { liftColor } from './scene3d.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const COL = {};
export function loadColours() {
  for (const k of ['text', 'dim', 'faint', 'line', 'line2', 'lift', 'est', 'dsp', 'warn', 'sink']) COL[k] = css(`--${k}`);
  COL.states = STATES.map((s) => css(`--s-${s.toLowerCase()}`));
  COL.mono = css('--mono');
}

/** Size a canvas for the device pixel ratio; returns a context scaled to CSS pixels. */
function fit(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}
const rgb = (c, a = 1) => `rgba(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0},${a})`;

// ── vertical air velocity, last 40 s ────────────────────────────────────────
const range = { lo: -1.5, hi: 3 };
export function drawStrip(cv, sim, ui) {
  const { g, w, h } = fit(cv);
  const H = sim.hist;
  const span = 40;
  const padL = 30;
  const pw = w - padL - 4;
  const tNow = sim.t;
  const n = Math.min(H.len, span * 20);
  const i0 = H.len - n;

  let lo = -1.2;
  let hi = 2.5;
  for (let i = i0; i < H.len; i++) {
    const j = H.idx(i);
    lo = Math.min(lo, H.wTrue[j] - 0.4, H.wKf[j] - 0.4);
    hi = Math.max(hi, H.wTrue[j] + 0.5, H.wKf[j] + 0.5);
  }
  range.lo += (lo - range.lo) * 0.08;
  range.hi += (hi - range.hi) * 0.08;
  const X = (t) => padL + pw * (1 - (tNow - t) / span);
  const Y = (v) => 6 + (h - 20) * (1 - (v - range.lo) / (range.hi - range.lo));

  g.font = `10px ${COL.mono}`;
  g.textBaseline = 'middle';
  const stepV = range.hi - range.lo > 7 ? 2 : 1;
  for (let v = Math.ceil(range.lo / stepV) * stepV; v <= range.hi; v += stepV) {
    g.strokeStyle = v === 0 ? COL.line2 : COL.line;
    g.beginPath();
    g.moveTo(padL, Y(v));
    g.lineTo(w, Y(v));
    g.stroke();
    g.fillStyle = COL.faint;
    g.textAlign = 'right';
    g.fillText((v > 0 ? '+' : '') + v, padL - 6, Y(v));
  }
  g.textAlign = 'center';
  g.textBaseline = 'alphabetic';
  g.fillStyle = COL.faint;
  for (let s = 10; s < span; s += 10) g.fillText(`−${s} s`, X(tNow - s), h - 2);
  if (n < 2) return;

  g.save();
  g.beginPath();
  g.rect(padL, 0, pw + 4, h - 12);
  g.clip();
  // Kalman ±2σ band
  g.beginPath();
  for (let i = i0; i < H.len; i++) {
    const j = H.idx(i);
    g.lineTo(X(H.t[j]), Y(H.wKf[j] + 2 * H.sig[j]));
  }
  for (let i = H.len - 1; i >= i0; i--) {
    const j = H.idx(i);
    g.lineTo(X(H.t[j]), Y(H.wKf[j] - 2 * H.sig[j]));
  }
  g.fillStyle = 'rgba(106,174,232,0.14)';
  g.fill();
  const line = (key, colour, lw) => {
    g.beginPath();
    for (let i = i0; i < H.len; i++) {
      const j = H.idx(i);
      const x = X(H.t[j]);
      const y = Y(H[key][j]);
      if (i === i0) g.moveTo(x, y);
      else g.lineTo(x, y);
    }
    g.strokeStyle = colour;
    g.lineWidth = lw;
    g.lineJoin = 'round';
    g.stroke();
  };
  line('wDsp', COL.dsp, 1.3);
  line('wKf', COL.est, 1.6);
  if (ui.truth) line('wTrue', '#ffffff', 1.6);
  g.restore();
}

// ── flight profile: altitude against distance along the route ───────────────
export function drawProfile(cv, sim, ui) {
  const { g, w, h } = fit(cv);
  const padL = 34;
  const pw = w - padL - 10;
  const top = 6;
  const ph = h - 24;
  const hMax = 600;
  const X = (x) => padL + (pw * x) / ROUTE.length;
  const Y = (v) => top + ph * (1 - v / hMax);
  const gd = sim.guidance;
  const gl = sim.glider;

  g.font = `10px ${COL.mono}`;
  g.textBaseline = 'middle';
  g.textAlign = 'right';
  for (const v of [0, 200, 400, 600]) {
    g.strokeStyle = COL.line;
    g.beginPath();
    g.moveTo(padL, Y(v));
    g.lineTo(padL + pw, Y(v));
    g.stroke();
    g.fillStyle = COL.faint;
    g.fillText(v, padL - 6, Y(v));
  }
  g.textAlign = 'center';
  g.textBaseline = 'alphabetic';
  for (let km = 0; km <= ROUTE.length / 1000; km += 2) {
    g.fillStyle = COL.faint;
    g.fillText(`${km} km`, Math.max(padL + 12, Math.min(padL + pw - 14, X(km * 1000))), h - 2);
  }

  g.save();
  g.beginPath();
  g.rect(padL, top, pw, ph);
  g.clip();
  // thermals (truth): one band per thermal, as wide as ±σ
  if (ui.truth)
    for (const th of sim.atm.thermals) {
      g.fillStyle = `rgba(216,162,74,${Math.min(0.5, 0.07 * th.wmax * th.s)})`;
      g.fillRect(X(th.x - th.sigma), Y(sim.atm.zi), Math.max(2, X(th.x + th.sigma) - X(th.x - th.sigma)), Y(0) - Y(sim.atm.zi));
    }
  // operating limits
  g.setLineDash([4, 4]);
  for (const [v, c] of [[LIMITS.ceiling, COL.faint], [LIMITS.hLow, COL.warn]]) {
    g.strokeStyle = c;
    g.beginPath();
    g.moveTo(padL, Y(v));
    g.lineTo(padL + pw, Y(v));
    g.stroke();
  }
  // final-glide path: altitude required to reach the destination from each point on the route
  g.strokeStyle = COL.est;
  g.beginPath();
  g.moveTo(X(ROUTE.length), Y(LIMITS.hArrive));
  g.lineTo(X(0), Y(LIMITS.hArrive + gd.glideSlope * ROUTE.length));
  g.stroke();
  g.setLineDash([]);

  // flown path, coloured by the vertical air velocity encountered
  const T = sim.trail;
  const c = [0, 0, 0];
  g.lineWidth = 1.6;
  for (let i = 1; i < T.len; i++) {
    const a = T.idx(i - 1);
    const b = T.idx(i);
    g.strokeStyle = ui.truth ? rgb(liftColor(T.w[b], c)) : '#ffffff';
    g.beginPath();
    g.moveTo(X(T.x[a]), Y(T.h[a]));
    g.lineTo(X(T.x[b]), Y(T.h[b]));
    g.stroke();
  }
  g.restore();

  g.textAlign = 'left';
  g.fillStyle = COL.faint;
  g.fillText('Operating ceiling', padL + 4, Y(LIMITS.ceiling) - 4);
  g.fillStyle = COL.warn;
  g.fillText('Motor engagement', padL + 4, Y(LIMITS.hLow) - 4);

  // destination and aircraft markers
  g.strokeStyle = COL.text;
  g.lineWidth = 1.2;
  g.beginPath();
  g.moveTo(X(ROUTE.length), Y(0));
  g.lineTo(X(ROUTE.length), Y(120));
  g.stroke();
  g.fillStyle = COL.text;
  g.beginPath();
  g.moveTo(X(ROUTE.length), Y(120));
  g.lineTo(X(ROUTE.length) - 9, Y(105));
  g.lineTo(X(ROUTE.length), Y(90));
  g.fill();
  g.fillStyle = COL.states[STATES.indexOf(gd.state)];
  g.strokeStyle = '#14171c';
  g.beginPath();
  g.arc(X(gl.x), Y(gl.h), 4.5, 0, 7);
  g.fill();
  g.stroke();
}

