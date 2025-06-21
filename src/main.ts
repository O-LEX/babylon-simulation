import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder, Mesh } from "@babylonjs/core";
import { ParticleSystem } from "./simulation";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
  const scene = new Scene(engine);

  const camera = new ArcRotateCamera(
    "camera",
    Math.PI / 2,
    Math.PI / 4,
    15,
    Vector3.Zero(),
    scene
  );
  camera.attachControl(canvas, true);

  const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);

  const ground = MeshBuilder.CreateGround("ground", { width: 20, height: 20 }, scene);
  ground.position.y = -5;

  const particleSystem = new ParticleSystem();
  particleSystem.createRandomParticles(10);

  // Create meshes for rendering particles
  const particleMeshes: Mesh[] = [];
  const positions = particleSystem.getParticlePositions();
  for (let i = 0; i < positions.length; i++) {
    const mesh = MeshBuilder.CreateSphere(`particle_${i}`, { diameter: 0.2 }, scene);
    mesh.position.copyFrom(positions[i]);
    particleMeshes.push(mesh);
  }

  const fpsDisplay = document.getElementById("fpsDisplay");

  scene.registerBeforeRender(() => {
    const deltaTime = engine.getDeltaTime() / 1000;
    
    if (fpsDisplay) {
      const fps = engine.getFps().toFixed(1);
      fpsDisplay.textContent = `FPS: ${fps}`;
    }
    
    particleSystem.update(deltaTime);
    
    // Update mesh positions
    const positions = particleSystem.getParticlePositions();
    for (let i = 0; i < positions.length && i < particleMeshes.length; i++) {
      particleMeshes[i].position.copyFrom(positions[i]);
    }
  });

  return scene;
};

function main() {
  const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
  const engine = new Engine(canvas, true);
  const scene = createScene(engine, canvas);
  engine.runRenderLoop(() => {
    scene.render();
  });

  window.addEventListener("resize", () => {
    engine.resize();
  });
}

main();