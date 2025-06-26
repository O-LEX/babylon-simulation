import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder, Mesh } from "@babylonjs/core";
// import { Solver, createCloth } from "./old";
import { ImplicitSolver, createChain, createCloth } from "./simulation";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
    const scene = new Scene(engine);

    const camera = new ArcRotateCamera("camera", Math.PI / 4, Math.PI / 3, 15, new Vector3(0, 2, 0), scene);
    camera.attachControl(canvas, true);

    const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);

    const fpsDisplay = document.getElementById("fpsDisplay");

    // const geometry = createChain(10, 100); // 10 vertices, stiffness=100
    const geometry = createCloth(6, 6, 8, 8, 100); // 6x6 world units, 8x8 resolution, stiffness=100
    const solver = new ImplicitSolver(geometry);

    // Create spheres to visualize nodes
    const sphereSize = 0.15; // Smaller spheres for cloth
    const spheres: Mesh[] = [];
    
    for (let i = 0; i < geometry.getNumVertices(); i++) {
        const sphere = MeshBuilder.CreateSphere(`sphere${i}`, { diameter: sphereSize }, scene);
        sphere.position = geometry.positions[i].clone();
        spheres.push(sphere);
    }

    // https://gafferongames.com/post/fix_your_timestep/
    let lastTime = performance.now();
    const fixedTimeStep = 1/60; // 60Hz physics
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
        
        const currentPositions = solver.getPositions();
        for (let i = 0; i < spheres.length; i++) {
            spheres[i].position.copyFrom(currentPositions[i]);
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
    const scene = createScene(engine, canvas);
    engine.runRenderLoop(() => {
        scene.render();
    });

    window.addEventListener("resize", () => {
        engine.resize();
    });
}

main();