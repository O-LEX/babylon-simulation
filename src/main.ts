import {
  Engine, Scene, ArcRotateCamera, Vector3,
  HemisphericLight, MeshBuilder, Mesh
} from "@babylonjs/core";

import { createCloth, createChain, createHighStiffnessRatioChain, createFixedCloth, createTwoCloths } from "./geometry";
import { Params } from "./params";
import { ImplicitSolver } from "./implicit";
import { VBDSolver } from "./vbd";
import { XPBDSolver } from "./xpbd";
import { AVBDSolver } from "./avbd";
import { ADMMSolver } from "./admm";

// Get simulation parameters from the UI
function getParamsFromUI(): Params {
  const gx = parseFloat((document.getElementById("gx") as HTMLInputElement).value);
  const gy = parseFloat((document.getElementById("gy") as HTMLInputElement).value);
  const gz = parseFloat((document.getElementById("gz") as HTMLInputElement).value);
  const dt = parseFloat((document.getElementById("dt") as HTMLInputElement).value);
  const numSubsteps = parseInt((document.getElementById("numSubsteps") as HTMLInputElement).value);
  const numIterations = parseInt((document.getElementById("numIterations") as HTMLInputElement).value);


  return {
    g: new Vector3(gx, gy, gz),
    dt,
    numSubsteps,
    numIterations
  };
}

// Create geometry from UI selection
function createGeometryFromUI() {
  const geometryType = (document.getElementById("geometry") as HTMLSelectElement).value;
  switch (geometryType) {
    case "cloth": return createCloth(5, 5, 10, 10, 1000, 1.0);
    case "chain": return createChain(5, 10, 1000, 1.0);
    case "fixed cloth": return createFixedCloth(5, 5, 10, 10, 1000, 1.0);
    case "two cloths": return createTwoCloths();
    case "highStiffnessRatioChain": return createHighStiffnessRatioChain();
    default: return createCloth(5, 5, 10, 10, 1000, 1.0);
  }
}

// Create solver from UI selection
function createSolverFromUI(geometry: any, params: Params) {
  const solverType = (document.getElementById("solver") as HTMLSelectElement).value;
  switch (solverType) {
    case "implicit": return new ImplicitSolver(geometry, params);
    case "vbd": return new VBDSolver(geometry, params);
    case "avbd": return new AVBDSolver(geometry, params);
    case "xpbd": return new XPBDSolver(geometry, params);
    case "admm": return new ADMMSolver(geometry, params);
    default: return new ADMMSolver(geometry, params);
  }
}

// Convert Float32Array to Vector3
function getVector3FromArray(arr: Float32Array, index: number): Vector3 {
  return new Vector3(arr[index * 3], arr[index * 3 + 1], arr[index * 3 + 2]);
}

// Create and return a Babylon.js scene
function createScene(engine: Engine, canvas: HTMLCanvasElement): Scene {
  const scene = new Scene(engine);

  const camera = new ArcRotateCamera("camera", Math.PI / 4, Math.PI / 3, 15, new Vector3(0, 2, 0), scene);
  camera.attachControl(canvas, true);
  new HemisphericLight("light", new Vector3(0, 1, 0), scene);

  const params = getParamsFromUI();
  const geometry = createGeometryFromUI();
  const solver = createSolverFromUI(geometry, params);

  const sphereSize = 0.1;
  const spheres: Mesh[] = [];
  let currentPositions = solver.pos;
  const numVertices = currentPositions.length / 3;

  // Create node spheres
  for (let i = 0; i < numVertices; i++) {
    const x = currentPositions[i * 3];
    const y = currentPositions[i * 3 + 1];
    const z = currentPositions[i * 3 + 2];

    const sphere = MeshBuilder.CreateSphere(`sphere${i}`, { diameter: sphereSize }, scene);
    sphere.position.set(x, y, z);
    spheres.push(sphere);
  }

  // Create edge lines
  const lines: Mesh[] = [];
  const currentEdges = solver.edges;
  const numEdges = currentEdges.length / 2;

  for (let i = 0; i < numEdges; i++) {
    const v0 = currentEdges[i * 2];
    const v1 = currentEdges[i * 2 + 1];
    const p0 = getVector3FromArray(currentPositions, v0);
    const p1 = getVector3FromArray(currentPositions, v1);

    const line = MeshBuilder.CreateLines(`line${i}`, {
      points: [p0, p1],
      updatable: true
    }, scene);

    lines.push(line);
  }


  const fpsDisplay = document.getElementById("fpsDisplay");

  scene.registerBeforeRender(() => {
    solver.step();

    // Update spheres
    currentPositions = solver.pos;
    for (let i = 0; i < spheres.length; i++) {
      spheres[i].position.set(
        currentPositions[i * 3],
        currentPositions[i * 3 + 1],
        currentPositions[i * 3 + 2]
      );
    }

    // Update lines
    for (let i = 0; i < numEdges; i++) {
      const v0 = currentEdges[i * 2];
      const v1 = currentEdges[i * 2 + 1];
      const p0 = getVector3FromArray(currentPositions, v0);
      const p1 = getVector3FromArray(currentPositions, v1);

      const lineData = [
        p0.x, p0.y, p0.z,
        p1.x, p1.y, p1.z
      ];
      lines[i].updateVerticesData("position", lineData);
    }

    if (fpsDisplay) {
      const fps = engine.getFps().toFixed(1);
      fpsDisplay.textContent = `FPS: ${fps}`;
    }
  });

  return scene;
}

// Main entry point
function main() {
  const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
  const engine = new Engine(canvas, true);
  let scene = createScene(engine, canvas);

  engine.runRenderLoop(() => {
    scene.render();
  });

  // Reset when buttons or selectors change
  const resetSimulation = document.getElementById("resetSimulation") as HTMLButtonElement;
  const geometrySelect = document.getElementById("geometry") as HTMLSelectElement;
  const solverSelect = document.getElementById("solver") as HTMLSelectElement;

  const reset = () => {
    scene.dispose();
    scene = createScene(engine, canvas);
  };

  resetSimulation.addEventListener("click", reset);
  geometrySelect.addEventListener("change", reset);
  solverSelect.addEventListener("change", reset);

  ["gx", "gy", "gz", "dt", "numSubsteps", "numIterations"].forEach(id => {
    const input = document.getElementById(id) as HTMLInputElement;
    input.addEventListener("change", reset);
  });

  window.addEventListener("resize", () => {
    engine.resize();
  });
}

main();
