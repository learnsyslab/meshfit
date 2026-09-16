// Interactive viewer for the frozen example scenes in docs/scenes/*.glb.
//
// Each scene holds the observed frame as a coloured point cloud, the fitted mesh
// placed in it, and where the generator's own pose would have put it, all in
// metres -- so point size is a real distance and the mesh is at its recovered
// size rather than normalised to fit the view.
//
// Markup:  <div class="scene-viewer" data-src="../scenes/<case>.glb"></div>

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const POINT_SIZE = 0.004;          // metres
const FRAME_DISTANCE = 2.6;        // in object radii
const GHOST = { color: 0x8c93a8, opacity: 0.22 };  // the initial pose

function control(label, checked, onChange) {
  const wrap = document.createElement('label');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = checked;
  box.addEventListener('change', () => onChange(box.checked));
  wrap.append(box, document.createTextNode(label));
  return wrap;
}

function build(container) {
  const canvas = document.createElement('div');
  canvas.className = 'scene-viewer__canvas';
  const status = document.createElement('div');
  status.className = 'scene-viewer__status';
  status.textContent = 'loading';
  const bar = document.createElement('div');
  bar.className = 'scene-viewer__controls';
  container.append(canvas, status, bar);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  canvas.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 2.0));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(1, 2, 1.5);
  scene.add(key);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = false;

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  return { scene, camera, controls, status, bar, resize };
}

function frame(view, mesh) {
  // Frame on the mesh rather than the whole cloud: the cloud is backdrop, and
  // fitting it in view would leave the object a speck in the middle.
  const box = new THREE.Box3().setFromObject(mesh);
  const centre = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2;

  view.camera.near = radius / 100;
  view.camera.far = radius * 100;
  view.controls.target.copy(centre);
  view.camera.position.copy(centre).add(
    new THREE.Vector3(1, 0.55, 1).normalize().multiplyScalar(radius * FRAME_DISTANCE));
  view.controls.minDistance = radius * 0.5;
  view.controls.maxDistance = radius * 20;
  view.controls.update();
  view.camera.updateProjectionMatrix();
}

function load(container) {
  const view = build(container);
  new GLTFLoader().load(
    container.dataset.src,
    (gltf) => {
      const mesh = gltf.scene.getObjectByName('mesh');
      const points = gltf.scene.getObjectByName('points');
      const init = gltf.scene.getObjectByName('init');
      if (points) {
        points.material = new THREE.PointsMaterial({
          size: POINT_SIZE, sizeAttenuation: true, vertexColors: true });
      }
      if (init) {
        // Drawn last and without writing depth, so the fitted mesh stays
        // visible through it however the two overlap.
        init.material = new THREE.MeshStandardMaterial({
          color: GHOST.color, opacity: GHOST.opacity, transparent: true,
          depthWrite: false, roughness: 0.8, side: THREE.DoubleSide });
        init.renderOrder = 1;
      }
      view.scene.add(gltf.scene);
      if (mesh) frame(view, mesh);
      view.resize();

      view.status.remove();
      if (mesh) view.bar.append(control('fitted', true, (on) => { mesh.visible = on; }));
      if (init) view.bar.append(control('init pose', true, (on) => { init.visible = on; }));
      if (points) view.bar.append(control('points', true, (on) => { points.visible = on; }));
      const reset = document.createElement('button');
      reset.textContent = 'reset view';
      reset.addEventListener('click', () => { if (mesh) frame(view, mesh); });
      view.bar.appendChild(reset);
    },
    (event) => {
      // Capped: a gzipped response reports decompressed bytes against a
      // compressed Content-Length, which walks past 100%.
      if (event.lengthComputable) {
        const done = Math.min(1, event.loaded / event.total);
        view.status.textContent = `loading ${Math.round(done * 100)}%`;
      }
    },
    () => { view.status.textContent = 'could not load this scene'; },
  );
}

// Deferred: each scene is a megabyte or two, and a page can hold several.
const watcher = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    watcher.unobserve(entry.target);
    load(entry.target);
  }
}, { rootMargin: '200px' });

for (const el of document.querySelectorAll('.scene-viewer')) watcher.observe(el);
