import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
renderer.setSize(canvas.clientWidth, canvas.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(65, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
camera.position.z = 4;


const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(1, 1, 1).normalize();
scene.add(light);

const loader = new GLTFLoader();
loader.load(
  '/static/3D_assets/Hand.glb', 
  function (gltf) {
    const model = gltf.scene;

    model.scale.set(0.5, 0.5, 0.5);
    model.position.set(0.9, -0.5, 0);

    scene.add(model);
    let forearmBone = null;

    model.traverse((child) => {
      if (child.isBone) {
        console.log(child.name); 
      }
    });

    model.traverse((child) => {
      if (child.isBone) {
        console.log(child.name); 
        if (child.name === 'forearmR') { 
          forearmBone = child;
        }
      }
    });

    if (!forearmBone) {
      console.error('Forearm bone not found!');
      return;
    }

    const clock = new THREE.Clock();

    function animate() {
      requestAnimationFrame(animate);

      const elapsed = clock.getElapsedTime();
      forearmBone.rotation.y = Math.sin(elapsed * 4) * 0.2; 

      renderer.render(scene, camera);
    }

    animate();
  },
  undefined,
  function (error) {
    console.error('An error occurred loading the model:', error);
  }
);

// Handle Resize
window.addEventListener('resize', () => {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
});
