import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder, Mesh } from "@babylonjs/core";
// import { Solver, createCloth } from "./old";
// import { ImplicitSolver, createChain, createCloth } from "./simulation";
import { DERSolver, createRod } from "./der";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
    const scene = new Scene(engine);

    const camera = new ArcRotateCamera("camera", Math.PI / 4, Math.PI / 3, 15, new Vector3(0, 2, 0), scene);
    camera.attachControl(canvas, true);

    const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);

    const fpsDisplay = document.getElementById("fpsDisplay");

    // Simple DER simulation - straight rod
    const derGeometry = createRod(10, 3.0, 1, 1, 1, 0.01, 0.1);
    const derSolver = new DERSolver(derGeometry);

    // Create spheres to visualize nodes
    const sphereSize = 0.1;
    const spheres: Mesh[] = [];

    let currentPositions = derSolver.getPositions();
    
    for (let i = 0; i < derGeometry.getNumVertices(); i++) {
        const sphere = MeshBuilder.CreateSphere(`sphere${i}`, { diameter: sphereSize }, scene);
        sphere.position = currentPositions[i];
        spheres.push(sphere);
    }

    let edges = MeshBuilder.CreateLines("edges", {
        points: currentPositions,
        updatable: true
    }, scene);

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
            derSolver.step(fixedTimeStep);
            accumulator -= fixedTimeStep;
        }
        
        currentPositions = derSolver.getPositions();
        for (let i = 0; i < spheres.length; i++) {
            spheres[i].position.copyFrom(currentPositions[i]);
        }
        edges.updateVerticesData("position", currentPositions.flatMap((v) => [v.x, v.y, v.z]));

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
    const scene = createScene(engine, canvas);
    engine.runRenderLoop(() => {
        scene.render();
    });

    window.addEventListener("resize", () => {
        engine.resize();
    });
}

main();