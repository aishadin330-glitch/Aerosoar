// Machine-learning pipeline window: network input, output against truth, error history, model comparison.
import { WIN, FEAT } from '../sim/sim.js';

const VIOLET = '#c9a6ff';
const CH = ['Rel. east', 'Rel. north', 'Vert. air vel.', 'Uncertainty', 'cos heading', 'sin heading', 'Airspeed', 'Wind east', 'Wind north'];
const RANGE = [3, 3, 2, 1, 1, 1, 0.5, 1, 1]; // normalised units mapped to the colour scale

function fit(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  if (!w || !h) return null;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}
const $ = (id) => document.getElementById(id);
const set = (id, t) => {
  const e = $(id);
  if (e && e.textContent !== t) e.textContent = t;
};

export class MLPanel {
  constructor(sim, ui, COL) {
    this.sim = sim;
    this.ui = ui;
    this.COL = COL;
    this.el = $('ml');
    this.boxText = '';
    $('mlToggle').addEventListener('click', () => this.show(this.el.hidden));
    $('mlClose').addEventListener('click', () => this.show(false));
    this.drag();
  }
  show(on) {
    this.el.hidden = !on;
    $('mlToggle').classList.toggle('on', on);
  }
  get open() {
    return !this.el.hidden;
  }
  drag() {
    const head = $('mlHead');
    head.addEventListener('pointerdown', (ev) => {
      if (ev.target.closest('button')) return;
      const r = this.el.getBoundingClientRect();
      const p = this.el.offsetParent.getBoundingClientRect();
      const dx = ev.clientX - r.left;
      const dy = ev.clientY - r.top;
      head.setPointerCapture(ev.pointerId);
      const move = (e) => {
        this.el.style.left = `${Math.max(0, Math.min(p.width - 80, e.clientX - p.left - dx))}px`;
        this.el.style.top = `${Math.max(0, Math.min(p.height - 40, e.clientY - p.top - dy))}px`;
        this.el.style.right = 'auto';
      };
      const up = () => {
        head.removeEventListener('pointermove', move);
        head.removeEventListener('pointerup', up);
      };
      head.addEventListener('pointermove', move);
      head.addEventListener('pointerup', up);
    });
  }
  /** Fill the static parts once the model and its reports are available. */
  setModel(model, summary) {
    const m = summary.models.find((x) => x.model === model.name) || {};
    const label = (n) => (n === 'lstm' ? 'LSTM' : 'Transformer');
    set('ml-name', label(model.name));
    this.boxText = `${label(model.name)}, 2 layers, ${(m.params / 1000).toFixed(0)}k parameters`;
    const f = (v, d = 1) => (v === undefined ? '–' : v.toFixed(d));
    const rows = summary.models
      .map((r) => `<tr class="${r.model === model.name ? 'sel' : ''}"><td>${label(r.model)}</td><td>${(r.params / 1000).toFixed(0)}k</td><td>${f(r.test.centre_err_m)}</td><td>${f(100 * r.test.coverage95, 0)} %</td><td>${f(r.ood.centre_err_m)}</td><td>${f(100 * r.ood.coverage95, 0)} %</td></tr>`)
      .join('');
    $('ml-sel').innerHTML = '<tr><th>Model</th><th>Params</th><th>Err, m</th><th>Cov.</th><th>OOD err, m</th><th>OOD cov.</th></tr>' + rows;
    set('ml-foot', `Trained on ${(summary.models[0].train_windows / 1000).toFixed(0)}k simulated windows. Err is the mean centre error on held-out flights; Cov. is the share of true centres inside the 95 % region. OOD: sink-ring thermals, stronger wind, higher noise.`);
  }
  update() {
    if (!this.open) return;
    const sim = this.sim;
    const o = sim.ml;
    $('ml-box').classList.toggle('hot', !!o && sim.t - o.t < 0.6);
    set('ml-box-s', this.boxText + (o ? `, ${o.ms.toFixed(1)} ms` : ''));
    this.drawInput();
    const g = sim.glider;
    const th = sim.atm.dominant(g.x, g.y, 250);
    const weff = th ? th.wmax * th.s * sim.atm.hprof(g.h) : 0;
    const real = th && weff > 0.5;
    set('ml-p', o ? `${(100 * o.p).toFixed(0)} %` : '–');
    set('ml-pt', real ? 'present' : 'absent');
    const on = o && o.p >= 0.3;
    set('ml-w', on ? `${o.w.toFixed(1)} ±${(2 * o.sdw).toFixed(1)} m/s` : '–');
    set('ml-wt', real ? weff.toFixed(1) : '–');
    set('ml-s', on ? `${o.s.toFixed(0)} ±${(2 * o.sds).toFixed(0)} m` : '–');
    set('ml-st', real ? th.sigma.toFixed(0) : '–');
    const e = on && real ? Math.hypot(o.x - th.x, o.y - th.y) : null;
    set('ml-e', e === null ? '–' : `${e.toFixed(0)} m`);
    this.drawErr();
  }
  drawInput() {
    const c = fit($('ml-in'));
    if (!c) return;
    const { g, w, h } = c;
    const win = this.sim.window();
    const padL = 86;
    const rows = FEAT.length;
    const cw = (w - padL) / WIN;
    const rh = h / rows;
    g.font = `10px ${this.COL.mono}`;
    g.textBaseline = 'middle';
    g.fillStyle = this.COL.faint;
    CH.forEach((n, r) => g.fillText(n, 0, r * rh + rh / 2));
    if (!win) {
      g.fillText('Collecting 16 s of history', padL + 8, h / 2);
      return;
    }
    for (let t = 0; t < WIN; t++)
      for (let r = 0; r < rows; r++) {
        const v = Math.max(-1, Math.min(1, win[t * rows + r] / RANGE[r]));
        g.fillStyle = v >= 0 ? `rgba(216,162,74,${(0.12 + 0.88 * v).toFixed(2)})` : `rgba(125,147,189,${(0.12 - 0.88 * v).toFixed(2)})`;
        g.fillRect(padL + t * cw, r * rh + 1, cw + 0.5, rh - 1);
      }
  }
  drawErr() {
    const c = fit($('ml-err'));
    if (!c) return;
    const { g, w, h } = c;
    const H = this.sim.mlHist;
    const padL = 28;
    const pw = w - padL - 4;
    const top = 4;
    const ph = h - 16;
    const yMax = 150;
    const span = 120;
    const tNow = this.sim.t;
    const X = (t) => padL + pw * (1 - (tNow - t) / span);
    const Y = (v) => top + ph * (1 - Math.min(v, yMax) / yMax);
    g.font = `10px ${this.COL.mono}`;
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    for (const v of [0, 50, 100, 150]) {
      g.strokeStyle = this.COL.line;
      g.beginPath();
      g.moveTo(padL, Y(v));
      g.lineTo(w, Y(v));
      g.stroke();
      g.fillStyle = this.COL.faint;
      g.fillText(v, padL - 5, Y(v));
    }
    g.textAlign = 'center';
    g.textBaseline = 'alphabetic';
    for (const s of [30, 60, 90]) g.fillText(`−${s} s`, X(tNow - s), h - 2);
    const line = (key, col) => {
      g.strokeStyle = col;
      g.lineWidth = 1.5;
      g.beginPath();
      let pen = false;
      for (let i = 0; i < H.len; i++) {
        const j = H.idx(i);
        const v = H[key][j];
        if (!(v === v) || H.t[j] < tNow - span) {
          pen = false;
          continue;
        }
        if (pen) g.lineTo(X(H.t[j]), Y(v));
        else {
          g.moveTo(X(H.t[j]), Y(v));
          pen = true;
        }
      }
      g.stroke();
    };
    line('errEst', this.COL.est);
    line('err', VIOLET);
  }
}
