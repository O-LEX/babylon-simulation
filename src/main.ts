import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder, Mesh } from "@babylonjs/core";
import { ImplicitSolver, createChain, createCloth } from "./simulation";
import { DERSolver, createRod, createLShapedRod } from "./der2";
// BSM (Block Sparse Matrix) test
import "./bsm";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
    const scene = new Scene(engine);

    const camera = new ArcRotateCamera("camera", Math.PI / 4, Math.PI / 3, 15, new Vector3(0, 2, 0), scene);
    camera.attachControl(canvas, true);

    const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);

    const fpsDisplay = document.getElementById("fpsDisplay");
    
    // Get simulation type from HTML select
    const simulationSelect = document.getElementById("simulationType") as HTMLSelectElement;
    const simulationType = simulationSelect.value;
    console.log("Selected simulation type:", simulationType);

    let solver: any;
    let geometry: any;
    let sphereSize = 0.1;

    // Create simulation based on selected type
    if (simulationType === "chain") {
        geometry = createChain(5, 10, 1000, 1.0);
        solver = new ImplicitSolver(geometry);
        sphereSize = 0.1;
    } else if (simulationType === "cloth") {
        geometry = createCloth(5, 5, 10, 10, 1000, 1.0);
        solver = new ImplicitSolver(geometry);
        sphereSize = 0.05;
    } else {
        // Default to DER rod
        // geometry = createRod(10, 3.0, 100, 0.001, 1, 0.01, 0.1);
        geometry = createLShapedRod(10, 3.0, 100, 0.1, 1, 0.01, 0.1);
        solver = new DERSolver(geometry);
        sphereSize = 0.05;
    }

    // Create spheres to visualize nodes
    const spheres: Mesh[] = [];
    let currentPositions = solver.getPositions();
    for (let i = 0; i < currentPositions.length; i++) {
        const sphere = MeshBuilder.CreateSphere(`sphere${i}`, { diameter: sphereSize }, scene);
        sphere.position = currentPositions[i];
        spheres.push(sphere);
    }

    let lines: Mesh[] = [];
    let currentEdges = solver.getEdges();
    for (let i = 0; i < currentEdges.length; i++) {
        const [v0, v1] = currentEdges[i];
        const line = MeshBuilder.CreateLines(`line${i}`, {
            points: [currentPositions[v0], currentPositions[v1]],
            updatable: true
        }, scene);
        lines.push(line);
    }

    // https://gafferongames.com/post/fix_your_timestep/
    let lastTime = performance.now();
    const fixedTimeStep = 1/60;
    let accumulator = 0;

    scene.registerBeforeRender(() => {
        const currentTime = performance.now();
        const frameTime = Math.min((currentTime - lastTime) / 1000, 0.25); // Cap at 250ms
        lastTime = currentTime;
        accumulator += frameTime;
        while (accumulator >= fixedTimeStep) {
            solver.step(fixedTimeStep);
            accumulator -= fixedTimeStep;
        }
        
        currentPositions = solver.getPositions();
        for (let i = 0; i < spheres.length; i++) {
            spheres[i].position.copyFrom(currentPositions[i]);
        }
        
        for (let i = 0; i < currentEdges.length; i++) {
            const [v0, v1] = currentEdges[i];
            const lineData = [
                currentPositions[v0].x, currentPositions[v0].y, currentPositions[v0].z,
                currentPositions[v1].x, currentPositions[v1].y, currentPositions[v1].z
            ];
            lines[i].updateVerticesData("position", lineData);
        }

        if (fpsDisplay) {
            const fps = engine.getFps().toFixed(1);
            fpsDisplay.textContent = `FPS: ${fps}`;
        }
    });

    return scene;
};

function main() {
    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    const engine = new Engine(canvas, true);
    let scene = createScene(engine, canvas);
    
    engine.runRenderLoop(() => {
        scene.render();
    });

    // Reset functionality
    const resetSimulation = document.getElementById("resetSimulation") as HTMLButtonElement;
    const simulationSelect = document.getElementById("simulationType") as HTMLSelectElement;
    
    const reset = () => {
        console.log("Resetting simulation...");
        
        // Dispose current scene
        scene.dispose();
        
        // Create new scene with current selection
        scene = createScene(engine, canvas);
    };
    
    resetSimulation.addEventListener("click", reset);
    
    // Auto-reset when simulation type changes
    simulationSelect.addEventListener("change", reset);

    window.addEventListener("resize", () => {
        engine.resize();
    });
}

main();