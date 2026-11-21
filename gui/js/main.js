// Interface wiring and the real-time loop.
import { Sim, SCENARIOS, STATES, ROUTE, DT, defaultConfig } from './sim/sim.js';
import { Scene3D } from './view/scene3d.js';
import { COL, loadColours, drawStrip, drawProfile } from './view/panels.js';
import { MLModel } from './sim/ml.js';
import { MLPanel } from './view/mlpanel.js';

const $ = (id) => document.getElementById(id);
const cfg = defaultConfig('calm');
const sim = new Sim(cfg);
const ui = { running: false, speed: 4, truth: true };
window.aerosoar = { sim, ui }; // exposed for inspection from the browser console

loadColours();
const scene = new Scene3D($('view'), $('labels'));
const mlPanel = new MLPanel(sim, ui, COL);

// trained network: weights and comparison report are produced by ml/train.py
fetch('ml/summary.json')
  .then((r) => r.json())
  .then(async (summary) => {
    const model = await MLModel.load(`ml/${summary.selected}.json`);
    sim.mlModel = model;
    mlPanel.setModel(model, summary);
    const b = $('est-ml');
    b.disabled = false;
    b.textContent = model.name === 'lstm' ? 'LSTM network' : 'Transformer';
    b.title = 'Trained neural network on a 16 s window of filtered measurements';
  })
  .catch(() => {});

// ── controls ────────────────────────────────────────────────────────────────
function choice(id, onPick) {
  const box = $(id);
  box.addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-v]');
    if (!b) return;
    for (const x of box.querySelectorAll('button')) x.classList.toggle('on', x === b);
    onPick(b.dataset.v);
  });
}
for (const [key, s] of Object.entries(SCENARIOS)) {
  const b = document.createElement('button');
  b.dataset.v = key;
  b.textContent = s.name;
  b.classList.toggle('on', key === cfg.scenario);
  $('scenarios').appendChild(b);
}
choice('scenarios', (v) => {
  cfg.scenario = v;
  $('scnNote').textContent = SCENARIOS[v].note;
  restart();
});
choice('speed', (v) => (ui.speed = Number(v)));
choice('estimator', (v) => {
  cfg.estimator = v;
  sim.setEstimator();
});
choice('camera', (v) => scene.setMode(v));

const noise = $('s-noise');
const showNoise = () => ($('o-noise').textContent = `× ${cfg.noise.toFixed(2)}`);
noise.value = cfg.noise;
noise.addEventListener('input', () => {
  cfg.noise = Number(noise.value);
  showNoise();
});
$('faults').addEventListener('click', (ev) => {
  const b = ev.target.closest('button');
  if (b) cfg.faults[b.dataset.f] = b.classList.toggle('on');
});
$('truth').addEventListener('change', (ev) => {
  ui.truth = ev.target.checked;
  document.body.classList.toggle('blind', !ui.truth);
});

function setRunning(on) {
  ui.running = on;
  const b = $('run');
  b.textContent = on ? 'Pause' : sim.done ? 'Restart' : sim.t > 0 ? 'Resume' : 'Start';
  b.classList.toggle('running', on);
}
function restart() {
  sim.reset();
  scene.first = true;
  setRunning(ui.running);
  refreshText(true);
}
function toggleRun() {
  if (sim.done) {
    restart();
    setRunning(true);
  } else setRunning(!ui.running);
}
$('run').addEventListener('click', toggleRun);
$('reset').addEventListener('click', () => {
  setRunning(false);
  restart();
});
window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Space') {
    ev.preventDefault();
    toggleRun();
  } else if (ev.code === 'KeyR') {
    setRunning(false);
    restart();
  }
});

// ── guidance-state strip ────────────────────────────────────────────────────
const LABEL = { CRUISE: 'Cruise', ASSESS: 'Assess thermal', CLIMB: 'Climb', MOTOR: 'Motor', ARRIVED: 'Arrived' };
const fsmEls = STATES.map((s, i) => {
  const li = document.createElement('li');
  li.textContent = LABEL[s];
  li.style.setProperty('--c', COL.states[i]);
  $('fsmStates').appendChild(li);
  return li;
});

// ── numerical read-outs (updated at 10 Hz) ──────────────────────────────────
const sgn = (v, d = 1) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const set = (id, text) => {
  const e = $(id);
  if (e.textContent !== text) e.textContent = text;
};
const NAMES = { pf: 'Particle filter', ekf: 'Extended Kalman', ml: 'Neural network', reactive: 'Variometer rule', oracle: 'Oracle' };
let lastText = 0;
function refreshText(force) {
  const now = performance.now();
  if (!force && now - lastText < 100) return;
  lastText = now;
  const g = sim.glider;
  const ch = sim.chain;
  const m = sim.m;
  const gd = sim.guidance;

  set('t-time', `${String(Math.floor(sim.t / 60)).padStart(2, '0')}:${String(Math.floor(sim.t % 60)).padStart(2, '0')}`);
  set('t-alt', ch.vkf.h.toFixed(0));
  set('t-climb', sgn(ch.vkf.vz));
  set('t-dist', (gd.dist / 1000).toFixed(1));
  const alert = $('b-alert');
  const msg = g.motor ? 'Motor on' : !sim.sensors.gnssOk ? 'GNSS unavailable' : '';
  alert.hidden = !msg;
  if (msg) set('b-alert', msg);

  const si = STATES.indexOf(gd.state);
  fsmEls.forEach((li, i) => li.classList.toggle('on', i === si));
  set('fsmReason', gd.reason);

  set('r-dsp', Math.sqrt(m.rmsDsp).toFixed(2));
  set('r-kf', Math.sqrt(m.rmsKf).toFixed(2));

  const e = sim.estimate();
  const pill = $('e-status');
  pill.textContent = e ? NAMES[cfg.estimator] : sim.trackActive ? `${NAMES[cfg.estimator]} (no model)` : 'No active track';
  pill.classList.toggle('live', !!e);
  const th = e ? sim.atm.dominant(g.x, g.y, 320) : null;
  set('e-w', e ? `${e.w.toFixed(1)} ±${(2 * e.sdW).toFixed(1)} m/s` : '–');
  set('e-s', e ? `${e.s.toFixed(0)} ±${(2 * e.sdS).toFixed(0)} m` : '–');
  set('e-p', e ? `±${(2 * e.posSd).toFixed(0)} m` : '–');
  set('e-wt', th ? (th.wmax * th.s * sim.atm.hprof(g.h)).toFixed(1) : '–');
  set('e-st', th ? th.sigma.toFixed(0) : '–');
  set('e-pt', e && m.ctrErr !== null ? `${m.ctrErr.toFixed(0)} m` : '–');
  const cov = m.coverN > 40 ? (100 * m.coverIn) / m.coverN : null;
  $('e-cov').style.width = cov === null ? '0' : `${cov}%`;
  $('e-cov').style.background = cov !== null && cov < 70 ? COL.warn : COL.est;
  set('e-covn', cov === null ? '–' : `${cov.toFixed(0)} %`);

  set('m-speed', sim.t > 5 ? `${((ROUTE.length - gd.dist) / sim.t).toFixed(1)} m/s` : '–');
  set('m-gain', `${m.gain.toFixed(0)} m`);
  set('m-used', `${m.used} / ${m.rejected + m.abandoned}`);
  set('m-wh', `${m.motorWh.toFixed(1)} Wh`);
}

// ── main loop ───────────────────────────────────────────────────────────────
let last = performance.now();
let acc = 0;
function frame(now) {
  const dtReal = Math.min(0.1, (now - last) / 1000);
  last = now;
  let simmed = 0;
  if (ui.running) {
    acc += dtReal * ui.speed;
    let n = 0;
    while (acc >= DT && n < 2400 && !sim.done) {
      sim.step();
      acc -= DT;
      n++;
    }
    if (n === 2400 || sim.done) acc = 0;
    simmed = n * DT;
    if (sim.done) setRunning(false);
  }
  scene.update(sim, ui, simmed);
  drawStrip($('c-strip'), sim, ui);
  drawProfile($('c-profile'), sim, ui);
  mlPanel.update();
  refreshText(false);
  requestAnimationFrame(frame);
}

new ResizeObserver(() => scene.resize()).observe($('stage'));
$('scnNote').textContent = SCENARIOS[cfg.scenario].note;
showNoise();
refreshText(true);
requestAnimationFrame(frame);
