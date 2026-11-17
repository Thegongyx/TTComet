/**
 * 首页 three.js 场景：两只小动物打乒乓球
 * =====================================
 * 纯程序化建模（球体/胶囊/圆锥拼），不依赖任何模型文件，打包体积只多 three.js 本身。
 *
 * 动画里复刻了产品高光视频的三个核心特效：
 *   1. 速度残影轨迹（越快的球拖尾越长、颜色越暖）
 *   2. 落点标记（球落台时环形扩散 + 中心点）
 *   3. 速度数值标签（跟随球浮动）
 *
 * 对外接口（挂在 window.TTCometScene）：
 *   setMode('full' | 'calm' | 'pause')  推理时降到低帧率，避免和 DirectML 抢 GPU
 *   shuffle()                           换一组对手
 */

import * as THREE from '/static/vendor/three.module.min.js';

/* ---------------- 动物外观档案 ---------------- */
const PROFILES = {
  panda: {
    name: '熊猫',
    body: 0xf3f5f8, belly: 0xffffff, dark: 0x23272f,
    ears: 'round', earColor: 0x23272f, eyePatch: true, muzzle: 0xffffff,
  },
  rabbit: {
    name: '小兔',
    body: 0xf7f0e7, belly: 0xfffaf3, dark: 0x3b2f2a,
    ears: 'long', earColor: 0xf7f0e7, earInner: 0xefb7c2, muzzle: 0xfffaf3,
  },
  penguin: {
    name: '企鹅',
    body: 0x2b3442, belly: 0xf4f7fb, dark: 0x11151c,
    ears: 'none', beak: 0xf2a33c, muzzle: 0xf4f7fb,
  },
  shiba: {
    name: '柴犬',
    body: 0xd98d4a, belly: 0xf7e8d6, dark: 0x2a211b,
    ears: 'pointy', earColor: 0xc97f3f, muzzle: 0xf7e8d6,
  },
};
const ORDER = ['panda', 'rabbit', 'penguin', 'shiba'];

/* ---------------- 常量 ---------------- */
const TABLE_TOP_Y = 0.558;
const HIT_Y = 0.80;
const HIT_X = 1.45;
const BOUNCE_X = 0.55;
const BOUNCE_P = 0.62;      // 单程中"落台"发生的进度点
const ROUND_SEC = 1.25;     // 单程时间
const TRAIL_N = 22;

/* ---------------- 状态 ---------------- */
let renderer, scene, camera;
let canvasEl, containerEl;
let animals = { left: null, right: null };
let ball, ballGlow, speedLabel;
let trail = [];              // { mesh, pos:Vector3, speed:number }
let marks = [];              // 落点环池
let dust;                    // 背景尘埃粒子
let dustPos, dustSpeed;
let gridHelper;
let blueLight, violetLight;
let tableGroup, netMesh;
let mode = 'full';
let rafId = null;
let lastFrame = 0;
let prevU = 0;
let prevP = 0;
let prevBallPos = new THREE.Vector3();
let swing = { left: 0, right: 0 };
let indexPair = [0, 1];
let switching = false;
let switchT = 0;
let labelTimer = 0;
let reducedMotion = false;
let labelCanvas, labelCtx, labelTex;

const FPS_CAP = { full: 45, calm: 15, pause: 0 };

/* ---------------- 工具 ---------------- */
const mat = (color, opts = {}) => new THREE.MeshStandardMaterial({
  color, roughness: opts.roughness ?? 0.8, metalness: opts.metalness ?? 0.05,
});

/** 速度归一化 -> 颜色（青蓝 -> 黄 -> 橙红，对应产品里的球速配色） */
const C_SLOW = new THREE.Color(0x3ec8ff);
const C_MID = new THREE.Color(0xffd166);
const C_FAST = new THREE.Color(0xff5a5f);
function speedColor(n, out) {
  if (n < 0.5) out.copy(C_SLOW).lerp(C_MID, n * 2);
  else out.copy(C_MID).lerp(C_FAST, (n - 0.5) * 2);
  return out;
}

/* ---------------- 动物 ---------------- */
function buildAnimal(key, side) {
  const p = PROFILES[key];
  const g = new THREE.Group();
  const bodyMat = mat(p.body);
  const bellyMat = mat(p.belly ?? p.body, { roughness: 0.9 });
  const darkMat = mat(p.dark, { roughness: 0.5 });

  const body = new THREE.Mesh(new THREE.SphereGeometry(0.36, 26, 18), bodyMat);
  body.scale.set(1, 1.06, 0.9);
  body.position.y = 0.45;
  g.add(body);

  const belly = new THREE.Mesh(new THREE.SphereGeometry(0.27, 22, 16), bellyMat);
  belly.scale.set(1, 1.05, 0.55);
  belly.position.set(0, 0.43, 0.22);
  g.add(belly);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.295, 26, 18), bodyMat);
  head.position.set(0, 0.92, 0.02);
  head.scale.set(1, 0.94, 0.98);
  g.add(head);

  if (p.ears === 'round') {
    const earMat = mat(p.earColor ?? p.body);
    for (const sx of [-1, 1]) {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.115, 16, 12), earMat);
      ear.position.set(sx * 0.19, 1.14, -0.02);
      ear.scale.set(1, 1, 0.7);
      g.add(ear);
    }
  } else if (p.ears === 'long') {
    const earMat = mat(p.earColor ?? p.body);
    const innerMat = mat(p.earInner ?? p.body);
    for (const sx of [-1, 1]) {
      const ear = new THREE.Mesh(new THREE.CapsuleGeometry(0.062, 0.3, 6, 12), earMat);
      ear.position.set(sx * 0.12, 1.27, -0.02);
      ear.rotation.z = sx * 0.16;
      g.add(ear);
      const inner = new THREE.Mesh(new THREE.CapsuleGeometry(0.032, 0.2, 6, 10), innerMat);
      inner.position.set(sx * 0.12, 1.27, 0.045);
      inner.rotation.z = sx * 0.16;
      g.add(inner);
    }
  } else if (p.ears === 'pointy') {
    const earMat = mat(p.earColor ?? p.body);
    for (const sx of [-1, 1]) {
      const ear = new THREE.Mesh(new THREE.ConeGeometry(0.095, 0.2, 14), earMat);
      ear.position.set(sx * 0.17, 1.16, -0.01);
      ear.rotation.z = sx * 0.22;
      g.add(ear);
    }
  }

  for (const sx of [-1, 1]) {
    if (p.eyePatch) {
      const patch = new THREE.Mesh(new THREE.SphereGeometry(0.085, 14, 10), darkMat);
      patch.scale.set(1, 1.15, 0.45);
      patch.position.set(sx * 0.105, 0.945, 0.225);
      g.add(patch);
    }
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.037, 12, 10), darkMat);
    eye.position.set(sx * 0.105, 0.945, 0.27);
    g.add(eye);
    const glint = new THREE.Mesh(
      new THREE.SphereGeometry(0.013, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xffffff }));
    glint.position.set(sx * 0.09, 0.96, 0.30);
    g.add(glint);
  }

  if (p.beak) {
    const beak = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.14, 12),
                                mat(p.beak, { roughness: 0.5 }));
    beak.rotation.x = Math.PI / 2;
    beak.position.set(0, 0.875, 0.29);
    g.add(beak);
  } else if (p.muzzle) {
    const mz = new THREE.Mesh(new THREE.SphereGeometry(0.105, 16, 12), mat(p.muzzle));
    mz.scale.set(1.15, 0.8, 0.7);
    mz.position.set(0, 0.858, 0.245);
    g.add(mz);
    const nose = new THREE.Mesh(new THREE.SphereGeometry(0.032, 10, 8), darkMat);
    nose.position.set(0, 0.885, 0.315);
    g.add(nose);
  }

  for (const sx of [-1, 1]) {
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.115, 14, 10), bellyMat);
    foot.scale.set(1, 0.6, 1.25);
    foot.position.set(sx * 0.16, 0.075, 0.06);
    g.add(foot);
  }

  const arms = {};
  const paddleSign = side === 'left' ? 1 : -1;
  for (const sx of [-1, 1]) {
    const shoulder = new THREE.Group();
    shoulder.position.set(sx * 0.33, 0.78, 0.16);

    const isPaddle = sx === paddleSign;
    const baseX = isPaddle ? -0.75 : 0.08;
    const baseZ = isPaddle ? sx * 0.16 : sx * 0.5;
    shoulder.rotation.set(baseX, 0, baseZ);

    const arm = new THREE.Mesh(new THREE.CapsuleGeometry(0.058, 0.26, 6, 12), bodyMat);
    arm.position.y = -0.15;
    shoulder.add(arm);

    if (isPaddle) {
      const paddle = new THREE.Group();
      paddle.position.set(0, -0.34, 0);
      const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.15, 8), darkMat);
      handle.position.y = 0.035;
      const blade = new THREE.Mesh(
        new THREE.CylinderGeometry(0.13, 0.13, 0.026, 24),
        mat(0xd9463f, { roughness: 0.5 }));
      blade.rotation.x = Math.PI / 2;
      blade.position.y = -0.085;
      paddle.add(handle, blade);
      shoulder.add(paddle);
    }
    g.add(shoulder);
    arms[sx] = { group: shoulder, baseX, baseZ, isPaddle };
  }

  g.rotation.y = side === 'left' ? Math.PI / 2 - 0.32 : -Math.PI / 2 + 0.32;
  g.position.x = side === 'left' ? -2.3 : 2.3;

  return { group: g, arms, paddleSign, key };
}

/* ---------------- 球台 ---------------- */
function buildTable() {
  const g = new THREE.Group();
  const topMat = mat(0x1c7a4f, { roughness: 0.72 });
  const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const metalMat = mat(0x3a4250, { roughness: 0.5, metalness: 0.4 });

  const top = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.07, 1.7), topMat);
  top.position.y = 0.52;
  g.add(top);

  const edgeGeoX = new THREE.BoxGeometry(3.0, 0.012, 0.03);
  const edgeGeoZ = new THREE.BoxGeometry(0.03, 0.012, 1.7);
  for (const z of [-0.845, 0.845]) {
    const e = new THREE.Mesh(edgeGeoX, lineMat);
    e.position.set(0, TABLE_TOP_Y, z);
    g.add(e);
  }
  for (const x of [-1.485, 1.485]) {
    const e = new THREE.Mesh(edgeGeoZ, lineMat);
    e.position.set(x, TABLE_TOP_Y, 0);
    g.add(e);
  }
  const mid = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.012, 1.7),
                             new THREE.MeshBasicMaterial({
                               color: 0xffffff, transparent: true, opacity: 0.35 }));
  mid.position.set(0, TABLE_TOP_Y, 0);
  g.add(mid);

  for (const x of [-1.3, 1.3]) {
    for (const z of [-0.65, 0.65]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.5, 10), metalMat);
      leg.position.set(x, 0.25, z);
      g.add(leg);
    }
  }

  const net = new THREE.Mesh(
    new THREE.BoxGeometry(0.02, 0.17, 1.78),
    new THREE.MeshStandardMaterial({
      color: 0xe8eef7, roughness: 0.9, transparent: true, opacity: 0.62,
      side: THREE.DoubleSide,
    }));
  net.position.set(0, 0.645, 0);
  g.add(net);
  netMesh = net;

  for (const z of [-0.89, 0.89]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.2, 8), metalMat);
    post.position.set(0, 0.63, z);
    g.add(post);
  }
  return g;
}

/* ---------------- 背景元素 ---------------- */
function buildBackground() {
  // 地面
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(16, 64),
    new THREE.MeshStandardMaterial({ color: 0x101a2b, roughness: 1 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.012;
  scene.add(floor);

  // 地面网格（低成本线条，制造纵深）
  gridHelper = new THREE.GridHelper(26, 26, 0x3b5080, 0x1e2a40);
  gridHelper.position.y = -0.006;
  gridHelper.material.transparent = true;
  gridHelper.material.opacity = 0.35;
  scene.add(gridHelper);

  // 浮动尘埃
  const N = 240;
  dustPos = new Float32Array(N * 3);
  dustSpeed = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    dustPos[i * 3] = (Math.random() - 0.5) * 18;
    dustPos[i * 3 + 1] = Math.random() * 5.5;
    dustPos[i * 3 + 2] = -Math.random() * 9 - 1;
    dustSpeed[i] = 0.12 + Math.random() * 0.35;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
  dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({
    color: 0x9fd0ff, size: 0.055, transparent: true, opacity: 0.5,
    depthWrite: false, sizeAttenuation: true,
  }));
  scene.add(dust);

  // 接地阴影
  const blob = (x, z, rx, rz, opacity) => {
    const m = new THREE.Mesh(
      new THREE.CircleGeometry(1, 32),
      new THREE.MeshBasicMaterial({
        color: 0x000000, transparent: true, opacity, depthWrite: false,
      }));
    m.rotation.x = -Math.PI / 2;
    m.scale.set(rx, rz, 1);
    m.position.set(x, -0.004, z);
    scene.add(m);
  };
  blob(0, 0, 1.95, 1.2, 0.3);
  blob(-2.3, 0.02, 0.44, 0.36, 0.38);
  blob(2.3, 0.02, 0.44, 0.36, 0.38);
}

/* ---------------- 落点标记池 ---------------- */
function buildMarks() {
  for (let i = 0; i < 6; i++) {
    const group = new THREE.Group();
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.075, 0.105, 28),
      new THREE.MeshBasicMaterial({
        color: 0x7fe3ff, transparent: true, opacity: 0,
        side: THREE.DoubleSide, depthWrite: false,
      }));
    ring.rotation.x = -Math.PI / 2;
    const dot = new THREE.Mesh(
      new THREE.CircleGeometry(0.028, 16),
      new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0, depthWrite: false,
      }));
    dot.rotation.x = -Math.PI / 2;
    group.add(ring, dot);
    group.position.y = TABLE_TOP_Y + 0.004;
    group.visible = false;
    scene.add(group);
    marks.push({ group, ring, dot, t: -1 });
  }
}

function spawnMark(x) {
  const m = marks.find((k) => k.t < 0) || marks[0];
  m.group.position.set(x, TABLE_TOP_Y + 0.004, 0);
  m.group.scale.setScalar(1);
  m.t = 0;
  m.group.visible = true;
}

/* ---------------- 速度标签 ---------------- */
function buildSpeedLabel() {
  labelCanvas = document.createElement('canvas');
  labelCanvas.width = 256;
  labelCanvas.height = 96;
  labelCtx = labelCanvas.getContext('2d');
  labelTex = new THREE.CanvasTexture(labelCanvas);
  labelTex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: labelTex, transparent: true, depthWrite: false, depthTest: false,
  }));
  sprite.scale.set(0.62, 0.2325, 1);
  sprite.position.set(0, 0, 0);
  scene.add(sprite);
  speedLabel = sprite;
}

function drawSpeedLabel(kmh) {
  const c = labelCtx;
  c.clearRect(0, 0, 256, 96);
  c.font = '600 52px "Segoe UI", "Microsoft YaHei", sans-serif';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillStyle = 'rgba(10,14,20,0.62)';
  const w = c.measureText(`${kmh} km/h`).width + 34;
  c.beginPath();
  if (c.roundRect) c.roundRect(128 - w / 2, 14, w, 68, 18);
  else c.rect(128 - w / 2, 14, w, 68);
  c.fill();
  c.fillStyle = '#e9f3ff';
  c.fillText(`${kmh} km/h`, 128, 50);
  labelTex.needsUpdate = true;
}

/* ---------------- 球的飞行轨迹 ---------------- */
function ballPos(p, dir) {
  const x0 = dir > 0 ? -HIT_X : HIT_X;
  const x1 = dir > 0 ? HIT_X : -HIT_X;
  const xb = dir > 0 ? BOUNCE_X : -BOUNCE_X;
  if (p < BOUNCE_P) {
    const t = p / BOUNCE_P;
    return {
      x: x0 + (xb - x0) * t,
      y: HIT_Y + (TABLE_TOP_Y - HIT_Y) * t + Math.sin(t * Math.PI) * 0.40,
    };
  }
  const t = (p - BOUNCE_P) / (1 - BOUNCE_P);
  return {
    x: xb + (x1 - xb) * t,
    y: TABLE_TOP_Y + (HIT_Y - TABLE_TOP_Y) * t + Math.sin(t * Math.PI) * 0.26,
  };
}

/* ---------------- 初始化 ---------------- */
function init(canvas) {
  canvasEl = canvas;
  containerEl = canvas.parentElement;

  renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: true, powerPreference: 'default',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x141f36, 9, 22);

  camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
  camera.position.set(0, 1.85, 5.35);
  camera.lookAt(0, 0.62, 0);

  scene.add(new THREE.HemisphereLight(0xbfd6ff, 0x1b2029, 1.0));
  const dir = new THREE.DirectionalLight(0xffffff, 1.35);
  dir.position.set(2.6, 4.2, 3.4);
  scene.add(dir);

  // 彩色补光：让暗部有层次，不显得"纯黑"
  blueLight = new THREE.PointLight(0x3b82f6, 9, 16, 2);
  blueLight.position.set(-3.6, 2.3, 2.4);
  scene.add(blueLight);
  violetLight = new THREE.PointLight(0x8b5cf6, 7, 16, 2);
  violetLight.position.set(3.6, 2.5, 1.8);
  scene.add(violetLight);

  tableGroup = buildTable();
  scene.add(tableGroup);

  buildBackground();
  buildMarks();
  buildSpeedLabel();

  // 球（带自发光，暗场里更醒目）
  ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.075, 20, 14),
    new THREE.MeshStandardMaterial({
      color: 0xffffff, roughness: 0.3, emissive: 0xffe6a8, emissiveIntensity: 0.55,
    }));
  scene.add(ball);

  // 球体外发光（sprite，零阴影开销）
  const glowCanvas = document.createElement('canvas');
  glowCanvas.width = glowCanvas.height = 128;
  const gctx = glowCanvas.getContext('2d');
  const grad = gctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,236,180,0.95)');
  grad.addColorStop(0.35, 'rgba(255,210,120,0.35)');
  grad.addColorStop(1, 'rgba(255,190,90,0)');
  gctx.fillStyle = grad;
  gctx.fillRect(0, 0, 128, 128);
  ballGlow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(glowCanvas),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  ballGlow.scale.set(0.62, 0.62, 1);
  scene.add(ballGlow);

  // 残影（速度着色 + 尺寸衰减）
  for (let i = 0; i < TRAIL_N; i++) {
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.052 * (1 - i * 0.035), 12, 10),
      new THREE.MeshBasicMaterial({
        color: 0x3ec8ff, transparent: true, opacity: 0, depthWrite: false,
      }));
    mesh.visible = false;
    scene.add(mesh);
    trail.push({ mesh, pos: new THREE.Vector3(), speed: 0 });
  }

  spawnPair(0, 1);
  resize();
  new ResizeObserver(resize).observe(containerEl);

  reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopLoop();
    else if (mode !== 'pause') startLoop();
  });

  prevBallPos.set(0, HIT_Y, 0);
  // 调试钩子（排查渲染问题用，不影响正常逻辑）
  window.__pp = {
    scene, camera, renderer, ball,
    get label() { return speedLabel; },
    get marks() { return marks; },
    get trail() { return trail; },
  };
  if (reducedMotion) { mode = 'pause'; renderOnce(0.6); }
  else startLoop();
}

function resize() {
  if (!renderer || !containerEl) return;
  const w = containerEl.clientWidth || 800;
  const h = containerEl.clientHeight || 252;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (mode === 'pause') renderOnce(0.6);
}

/* ---------------- 动物替换 ---------------- */
function spawnPair(i, j) {
  removeAnimal('left');
  removeAnimal('right');
  const leftKey = ORDER[i % ORDER.length];
  const rightKey = ORDER[j % ORDER.length];
  animals.left = buildAnimal(leftKey, 'left');
  animals.right = buildAnimal(rightKey, 'right');
  scene.add(animals.left.group);
  scene.add(animals.right.group);
  try {
    window.dispatchEvent(new CustomEvent('pp:pair', {
      detail: { left: leftKey, right: rightKey },   // 传 key，由前端按语言翻译
    }));
  } catch (e) { /* 事件失败不影响渲染 */ }
}

function removeAnimal(side) {
  const a = animals[side];
  if (!a) return;
  scene.remove(a.group);
  a.group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose?.();
  });
  animals[side] = null;
}

function applyPair() {
  const n = indexPair[0] + indexPair[1] + 1;
  indexPair = [n % ORDER.length, (n + 2) % ORDER.length];
  spawnPair(indexPair[0], indexPair[1]);
}

function shuffle() {
  const idle = rafId === null || document.hidden;
  if (switching) {
    if (!idle) return;          // 正在做缩放过场，忽略连点
    switching = false;          // 循环停了就别卡在中间态
    switchT = 0;
  }
  if (idle) {
    // 后台/暂停时没有动画可看，直接换并渲染一帧
    applyPair();
    renderOnce(0.6);
    return;
  }
  switching = true;
  switchT = 0;
}

/* ---------------- 循环 ---------------- */
function startLoop() {
  if (rafId !== null) return;
  lastFrame = 0;
  const loop = (ts) => {
    rafId = requestAnimationFrame(loop);
    const cap = FPS_CAP[mode] || 0;
    if (cap > 0 && ts - lastFrame < 1000 / cap) return;
    lastFrame = ts;
    renderOnce(ts / 1000, true);
  };
  rafId = requestAnimationFrame(loop);
}

function stopLoop() {
  if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
}

function renderOnce(t, animated) {
  if (!renderer) return;
  const tmpColor = new THREE.Color();

  if (animated) {
    // 换动物的缩放过场
    if (switching) {
      switchT += 1 / 45;
      const k = Math.min(1, switchT / 0.22);
      for (const side of ['left', 'right']) {
        if (animals[side]) animals[side].group.scale.setScalar(1 - 0.85 * k);
      }
      if (k >= 1) {
        applyPair();
        switching = false;
        switchT = 0;
      }
    } else if (animals.left && animals.right) {
      const pulse = 1 + Math.sin(t * 4) * 0.01;
      animals.left.group.scale.setScalar(pulse);
      animals.right.group.scale.setScalar(pulse);
    }

    // 呼吸
    for (const a of [animals.left, animals.right]) {
      if (!a) continue;
      a.group.position.y = Math.sin(t * 2.1 + (a.key === 'rabbit' ? 1 : 0)) * 0.012;
    }

    // ---- 球飞行 ----
    const u = (t / ROUND_SEC) % 2;          // 0..2，一个来回
    const dir = u < 1 ? 1 : -1;
    const p = u < 1 ? u : u - 1;            // 0..1 单程进度
    const bp = ballPos(p, dir);
    const z = Math.sin(t * 2.2) * 0.05;
    ball.position.set(bp.x, bp.y, z);
    ballGlow.position.copy(ball.position);

    // 速度（相邻帧位移，用于残影着色）
    const step = Math.hypot(bp.x - prevBallPos.x, bp.y - prevBallPos.y);
    const speedNorm = Math.min(1, step / 0.062);

    // ---- 落台判定（p 跨过 BOUNCE_P；两个方向都适用）----
    if (!(u < prevU) && prevP < BOUNCE_P && p >= BOUNCE_P) {
      spawnMark(dir > 0 ? BOUNCE_X : -BOUNCE_X);
    }
    // ---- 击球判定：单程切换 ----
    if (Math.floor(u) !== Math.floor(prevU)) {
      const side = u < 1 ? 'left' : 'right';
      swing[side] = 0.001;
    }
    prevU = u;
    prevP = p;
    prevBallPos.set(bp.x, bp.y, z);

    // ---- 残影 ----
    for (let i = trail.length - 1; i > 0; i--) {
      trail[i].pos.copy(trail[i - 1].pos);
      trail[i].speed = trail[i - 1].speed;
    }
    trail[0].pos.copy(ball.position);
    trail[0].speed = speedNorm;

    for (let i = 0; i < trail.length; i++) {
      const it = trail[i];
      const fade = 1 - i / trail.length;
      it.mesh.position.copy(it.pos);
      speedColor(it.speed, tmpColor);
      it.mesh.material.color.copy(tmpColor);
      it.mesh.material.opacity = 0.62 * Math.pow(fade, 1.4);
      it.mesh.scale.setScalar(0.5 + it.speed * 0.85);
      it.mesh.visible = i > 0;
    }

    // ---- 速度标签（每 0.12s 更新一次，避免频繁重绘 canvas）----
    labelTimer += 1 / 45;
    if (labelTimer > 0.12) {
      labelTimer = 0;
      drawSpeedLabel(Math.round(28 + speedNorm * 82));
    }
    speedLabel.position.set(ball.position.x, ball.position.y + 0.34, ball.position.z);
    speedLabel.material.opacity = 0.92;

    // ---- 落点环动画 ----
    for (const m of marks) {
      if (m.t < 0) continue;
      m.t += 1 / 45;
      const k = m.t / 0.75;
      if (k >= 1) {
        m.t = -1;
        m.group.visible = false;
        continue;
      }
      m.ring.scale.setScalar(1 + k * 4.2);
      m.ring.material.opacity = 0.85 * (1 - k);
      m.dot.material.opacity = 0.9 * Math.max(0, 1 - k * 2.2);
    }

    // ---- 挥拍 ----
    for (const side of ['left', 'right']) {
      const a = animals[side];
      if (!a) continue;
      for (const sx of [-1, 1]) {
        const arm = a.arms[sx];
        if (!arm) continue;
        if (arm.isPaddle && swing[side] > 0) {
          swing[side] = Math.min(1, swing[side] + 1 / 12);
          const sp = swing[side];
          arm.group.rotation.x = arm.baseX - Math.sin(sp * Math.PI) * 0.8;
          arm.group.rotation.z = arm.baseZ * (1 - sp * 0.4);
          if (sp >= 1) swing[side] = 0;
        } else {
          arm.group.rotation.x += (arm.baseX - arm.group.rotation.x) * 0.12;
          arm.group.rotation.z += (arm.baseZ - arm.group.rotation.z) * 0.12;
        }
      }
    }

    // ---- 背景：尘埃上浮 + 彩色补光游走 ----
    for (let i = 0; i < dustSpeed.length; i++) {
      dustPos[i * 3 + 1] += dustSpeed[i] * 0.008;
      if (dustPos[i * 3 + 1] > 5.8) {
        dustPos[i * 3 + 1] = -0.2;
        dustPos[i * 3] = (Math.random() - 0.5) * 18;
        dustPos[i * 3 + 2] = -Math.random() * 9 - 1;
      }
    }
    dust.geometry.attributes.position.needsUpdate = true;
    dust.rotation.y = Math.sin(t * 0.08) * 0.05;

    blueLight.position.x = -3.6 + Math.sin(t * 0.5) * 0.8;
    violetLight.position.x = 3.6 + Math.cos(t * 0.42) * 0.8;

    // ---- 相机轻微环绕 ----
    const ang = Math.sin(t * 0.22) * 0.16;
    camera.position.x = Math.sin(ang) * 5.35;
    camera.position.z = Math.cos(ang) * 5.35;
    camera.position.y = 1.85 + Math.sin(t * 0.37) * 0.05;
    camera.lookAt(0, 0.62, 0);

    if (netMesh) netMesh.material.opacity = 0.58 + Math.sin(t * 6) * 0.05;
  }

  renderer.render(scene, camera);
}

/* ---------------- 对外接口 ---------------- */
function setMode(next) {
  if (!['full', 'calm', 'pause'].includes(next)) return;
  const prev = mode;
  mode = next;
  if (next === 'pause') {
    stopLoop();
    renderOnce(0.6);
  } else if (prev === 'pause' || rafId === null) {
    startLoop();
  }
}

function boot() {
  const canvas = document.getElementById('ppScene');
  if (!canvas) return;
  try {
    init(canvas);
  } catch (e) {
    console.warn('[scene] WebGL 初始化失败，已隐藏动画区', e);
    canvas.parentElement?.classList.add('scene-fallback');
  }
}

window.TTCometScene = { setMode, shuffle };

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
