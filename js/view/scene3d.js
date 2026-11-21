// Three-dimensional flight view. World (east, north, up) maps to three.js (x, −z, y).
import * as THREE from 'three';
import { OrbitControls } from '../../vendor/OrbitControls.js';
import { LIMITS, ROUTE } from '../sim/sim.js';

const C = {
  ground: 0x171a20, grid1: 0x3a424e, grid2: 0x23282f, route: 0x7a8494,
  lift: 0xd8a24a, est: 0x6aaee8, craft: 0xe6e9ee, drop: 0x5a6472,
};
const GLIDER_SCALE = 14; // drawn larger than life so that it remains visible
const MAX_TH = 16;
const NP = 200; // particles per thermal column

/** steel blue (sink) → grey (still air) → ochre → pale yellow (strong lift) */
export function liftColor(w, out) {
  const stops = [[-2.5, 0.42, 0.52, 0.78], [0, 0.45, 0.48, 0.53], [1.6, 0.85, 0.64, 0.29], [4, 0.95, 0.86, 0.62]];
  let i = 0;
  while (i < stops.length - 2 && w > stops[i + 1][0]) i++;
  const a = stops[i];
  const b = stops[i + 1];
  const t = Math.min(1, Math.max(0, (w - a[0]) / (b[0] - a[0])));
  out[0] = a[1] + (b[1] - a[1]) * t;
  out[1] = a[2] + (b[2] - a[2]) * t;
  out[2] = a[3] + (b[3] - a[3]) * t;
  return out;
}

function softSprite() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)');
  gr.addColorStop(0.4, 'rgba(255,255,255,0.35)');
  gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function skyTexture() {
  const c = document.createElement('canvas');
  c.width = 4;
  c.height = 256;
  const g = c.getContext('2d');
  const gr = g.createLinearGradient(0, 0, 0, 256);
  gr.addColorStop(0, '#0e1014');
  gr.addColorStop(0.55, '#15181d');
  gr.addColorStop(0.78, '#262b33');
  gr.addColorStop(1, '#171a20');
  g.fillStyle = gr;
  g.fillRect(0, 0, 4, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function unitCircle(n = 72) {
  const pts = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a), 0, Math.sin(a)));
  }
  return new THREE.BufferGeometry().setFromPoints(pts);
}

function segment(colour, opacity = 1) {
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 1, 0)]),
    new THREE.LineBasicMaterial({ color: colour, transparent: opacity < 1, opacity })
  );
  line.frustumCulled = false;
  return line;
}

function buildGlider() {
  const g = new THREE.Group();
  const body = new THREE.MeshStandardMaterial({ color: C.craft, roughness: 0.5, metalness: 0.1 });
  const accent = new THREE.MeshStandardMaterial({ color: C.est, roughness: 0.5 });
  const fus = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.02, 1.25, 12), body);
  fus.rotation.z = -Math.PI / 2; // long axis → +x (nose)
  fus.position.x = -0.1;
  g.add(fus);
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.055, 12, 10), body);
  nose.position.x = 0.525;
  nose.scale.set(2.2, 1, 1);
  g.add(nose);
  for (const side of [1, -1]) {
    const wing = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.014, 1.0), body);
    wing.position.set(0.12, 0.065, side * 0.5);
    wing.rotation.x = -side * 0.07; // dihedral
    g.add(wing);
    const tip = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.016, 0.16), accent);
    tip.position.set(0.11, 0.105, side * 1.0);
    tip.rotation.x = -side * 0.07;
    g.add(tip);
  }
  const tail = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.01, 0.56), body);
  tail.position.set(-0.68, 0.2, 0);
  g.add(tail);
  const fin = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.22, 0.012), accent);
  fin.position.set(-0.66, 0.1, 0);
  g.add(fin);
  g.scale.setScalar(GLIDER_SCALE);
  return g;
}

export class Scene3D {
  constructor(canvas, labelLayer) {
    this.canvas = canvas;
    this.labelLayer = labelLayer;
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = (this.scene = new THREE.Scene());
    scene.background = skyTexture();
    scene.fog = new THREE.Fog(C.ground, 3500, 16000);
    this.camera = new THREE.PerspectiveCamera(48, 1, 2, 40000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 40;
    this.controls.maxDistance = 14000;
    this.controls.enablePan = false;
    this.target = new THREE.Vector3();
    this.mode = 'chase';

    scene.add(new THREE.HemisphereLight(0xdfe4ec, 0x14171c, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 2.0);
    sun.position.set(-300, 600, 250);
    scene.add(sun);

    const mid = ROUTE.length / 2;
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(60000, 60000), new THREE.MeshBasicMaterial({ color: C.ground }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.x = mid;
    scene.add(ground);
    const grid = new THREE.GridHelper(16000, 80, C.grid1, C.grid2); // 200 m cells
    grid.position.set(mid, 0.4, 0);
    scene.add(grid);

    // direct route and destination
    const route = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0.8, 0), new THREE.Vector3(ROUTE.length, 0.8, 0)]),
      new THREE.LineDashedMaterial({ color: C.route, dashSize: 60, gapSize: 40 })
    );
    route.computeLineDistances();
    scene.add(route);
    const circ = unitCircle(96);
    const destRing = new THREE.LineLoop(circ, new THREE.LineBasicMaterial({ color: C.craft }));
    destRing.scale.setScalar(LIMITS.arriveRadius);
    destRing.position.set(ROUTE.length, 0.9, 0);
    scene.add(destRing);
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 2.5, 260, 8), new THREE.MeshBasicMaterial({ color: C.craft }));
    pole.position.set(ROUTE.length, 130, 0);
    scene.add(pole);
    this.destLabel = this.makeLabel('tag dest');
    this.destLabel.textContent = 'Destination';

    // glider, height reference line, ground marker
    this.glider = buildGlider();
    scene.add(this.glider);
    this.drop = segment(C.drop, 0.7);
    scene.add(this.drop);
    this.foot = new THREE.Mesh(new THREE.RingGeometry(7, 9, 32), new THREE.MeshBasicMaterial({ color: C.drop, side: THREE.DoubleSide }));
    this.foot.rotation.x = -Math.PI / 2;
    scene.add(this.foot);

    // flown path, coloured by the vertical air velocity encountered
    this.trailMax = 6001;
    const tg = new THREE.BufferGeometry();
    tg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.trailMax * 3), 3));
    tg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.trailMax * 3), 3));
    this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ vertexColors: true }));
    this.trail.frustumCulled = false;
    scene.add(this.trail);

    // thermals (ground truth): rising particles and a footprint of radius σ
    const sprite = softSprite();
    this.th = [];
    for (let i = 0; i < MAX_TH; i++) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3), 3));
      const pts = new THREE.Points(
        geo,
        new THREE.PointsMaterial({ size: 13, map: sprite, color: C.lift, transparent: true, opacity: 0.3, depthWrite: false, blending: THREE.AdditiveBlending })
      );
      pts.frustumCulled = false;
      const ring = new THREE.LineLoop(circ, new THREE.LineBasicMaterial({ color: C.lift, transparent: true, opacity: 0.8 }));
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(1, 48),
        new THREE.MeshBasicMaterial({ color: C.lift, transparent: true, opacity: 0.09, depthWrite: false, blending: THREE.AdditiveBlending })
      );
      disc.rotation.x = -Math.PI / 2;
      const seed = new Float32Array(NP * 3); // radius factor, angle, height fraction
      for (let p = 0; p < NP; p++) {
        seed[p * 3] = Math.sqrt(-2 * Math.log(1 - Math.random() * 0.98));
        seed[p * 3 + 1] = Math.random() * Math.PI * 2;
        seed[p * 3 + 2] = Math.random();
      }
      scene.add(pts, ring, disc);
      this.th.push({ pts, ring, disc, seed });
    }

    // thermal estimate: 95 % confidence ellipse at flight level, core radius on the ground
    this.estEllipse = new THREE.LineLoop(circ, new THREE.LineBasicMaterial({ color: C.est }));
    const fill = new THREE.Mesh(
      new THREE.CircleGeometry(1, 64),
      new THREE.MeshBasicMaterial({ color: C.est, transparent: true, opacity: 0.14, depthWrite: false, side: THREE.DoubleSide })
    );
    fill.rotation.x = -Math.PI / 2;
    this.estFill = new THREE.Group();
    this.estFill.add(fill);
    this.estCore = new THREE.LineLoop(circ, new THREE.LineBasicMaterial({ color: C.est, transparent: true, opacity: 0.55 }));
    this.estStem = segment(C.est, 0.6);
    this.estGroup = new THREE.Group();
    this.estGroup.add(this.estEllipse, this.estFill, this.estCore, this.estStem);
    scene.add(this.estGroup);
    this.estLabel = this.makeLabel('tag est');

    this.v = new THREE.Vector3();
    this.col = [0, 0, 0];
    this.first = true;
    this.resize();
  }
  makeLabel(cls) {
    const d = document.createElement('div');
    d.className = cls;
    d.style.display = 'none';
    this.labelLayer.appendChild(d);
    return d;
  }
  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.w = w;
    this.h = h;
  }
  setMode(mode) {
    this.mode = mode;
    this.first = true;
  }
  place(label, x, y, z, text) {
    const v = this.v.set(x, y, z).project(this.camera);
    if (v.z > 1 || v.z < -1 || Math.abs(v.x) > 1.1 || Math.abs(v.y) > 1.1) {
      label.style.display = 'none';
      return;
    }
    label.style.display = '';
    label.style.left = `${(v.x * 0.5 + 0.5) * this.w}px`;
    label.style.top = `${(-v.y * 0.5 + 0.5) * this.h - 6}px`;
    if (text !== undefined && label.textContent !== text) label.textContent = text;
  }
  /** dtVis: seconds of simulated time since the previous frame (drives particle motion). */
  update(sim, ui, dtVis) {
    const g = sim.glider;
    const gp = this.glider.position.set(g.x, g.h, -g.y);
    this.glider.rotation.set(-g.phi, g.chi, g.gam, 'YZX');

    // camera: follows the aircraft ("chase") or shows the whole route
    const cam = this.camera;
    if (this.mode === 'route') {
      if (this.first) {
        this.target.set(ROUTE.length / 2, 150, 0);
        cam.position.set(ROUTE.length / 2, 1500, 7200);
      }
    } else {
      const d = this.v.copy(gp).sub(this.target);
      if (this.first) {
        this.target.copy(gp);
        cam.position.set(gp.x - 200, gp.y + 105, gp.z + 150);
      } else {
        this.target.add(d);
        cam.position.add(d);
      }
    }
    this.first = false;
    this.controls.target.copy(this.target);
    this.controls.update();

    const dp = this.drop.geometry.attributes.position;
    dp.setXYZ(0, g.x, 0.5, -g.y);
    dp.setXYZ(1, g.x, g.h, -g.y);
    dp.needsUpdate = true;
    this.foot.position.set(g.x, 0.9, -g.y);

    // flown path
    const tr = sim.trail;
    const pos = this.trail.geometry.attributes.position;
    const col = this.trail.geometry.attributes.color;
    const c = this.col;
    for (let i = 0; i < tr.len; i++) {
      const j = tr.idx(i);
      pos.setXYZ(i, tr.x[j], tr.h[j], -tr.y[j]);
      if (ui.truth) liftColor(tr.w[j], c);
      else c[0] = c[1] = c[2] = 0.8;
      col.setXYZ(i, c[0], c[1], c[2]);
    }
    pos.setXYZ(tr.len, g.x, g.h, -g.y);
    col.setXYZ(tr.len, c[0], c[1], c[2]);
    pos.needsUpdate = col.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, tr.len + 1);

    // thermals (ground truth)
    const atm = sim.atm;
    for (let i = 0; i < MAX_TH; i++) {
      const o = this.th[i];
      const th = atm.thermals[i];
      const vis = !!th && ui.truth;
      o.pts.visible = o.ring.visible = o.disc.visible = vis;
      if (!vis) continue;
      const p = o.pts.geometry.attributes.position;
      const rise = (2.5 + 5 * th.wmax * th.s) * dtVis;
      for (let k = 0; k < NP; k++) {
        let z = o.seed[k * 3 + 2] + rise / atm.zi;
        if (z > 1) {
          z -= 1;
          o.seed[k * 3 + 1] = Math.random() * Math.PI * 2;
        }
        o.seed[k * 3 + 2] = z;
        const rr = o.seed[k * 3] * th.sigma * (0.75 + 0.5 * z);
        const a = o.seed[k * 3 + 1] + z * 2.2;
        p.setXYZ(k, th.x + rr * Math.cos(a), z * atm.zi, -(th.y + rr * Math.sin(a)));
      }
      p.needsUpdate = true;
      o.pts.material.opacity = 0.07 + 0.06 * th.wmax * th.s;
      o.ring.position.set(th.x, 1, -th.y);
      o.ring.scale.setScalar(th.sigma);
      o.disc.position.set(th.x, 0.7, -th.y);
      o.disc.scale.setScalar(th.sigma * 1.6);
    }

    // thermal estimate
    const e = sim.estimate();
    this.estGroup.visible = !!e;
    if (e) {
      const half = (e.Pxx + e.Pyy) / 2;
      const dd = Math.sqrt(Math.max(0, (e.Pxx - e.Pyy) ** 2 / 4 + e.Pxy * e.Pxy));
      const a = Math.sqrt(5.99 * Math.max(1, half + dd));
      const b = Math.sqrt(5.99 * Math.max(1, half - dd));
      const ang = 0.5 * Math.atan2(2 * e.Pxy, e.Pxx - e.Pyy);
      for (const o of [this.estEllipse, this.estFill]) {
        o.position.set(e.x, g.h, -e.y);
        o.scale.set(a, 1, b);
        o.rotation.y = ang;
      }
      this.estCore.position.set(e.x, 1.2, -e.y);
      this.estCore.scale.setScalar(e.s);
      const sp = this.estStem.geometry.attributes.position;
      sp.setXYZ(0, e.x, 1, -e.y);
      sp.setXYZ(1, e.x, g.h, -e.y);
      sp.needsUpdate = true;
      this.place(this.estLabel, e.x, g.h + 30, -e.y, `Estimate  ${e.w.toFixed(1)} m/s  ±${e.posSd.toFixed(0)} m`);
    } else this.estLabel.style.display = 'none';

    this.place(this.destLabel, ROUTE.length, 280, 0);
    this.renderer.render(this.scene, cam);
  }
}
