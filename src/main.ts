import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder, Mesh, StandardMaterial, Color3 } from "@babylonjs/core";
import { Solver, createCloth } from "./simulation";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
    const scene = new Scene(engine);

    const camera = new ArcRotateCamera("camera", Math.PI / 2, Math.PI / 4, 10, new Vector3(0, 1, 0), scene);
    camera.attachControl(canvas, true);

    const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);

    // const ground = MeshBuilder.CreateGround("ground", { width: 20, height: 20 }, scene);

    const fpsDisplay = document.getElementById("fpsDisplay");

    // Create cloth simulation
    const cloth = createCloth(4, 3, 8, 6); // width=4, height=3, segmentsX=8, segmentsY=6
    const solver = new Solver(cloth);

    // Create spheres to visualize cloth nodes
    const sphereSize = 0.1;
    const spheres: Mesh[] = [];
    const material = new StandardMaterial("sphereMaterial", scene);
    material.diffuseColor = new Color3(0.0, 0.0, 0.0);
    
    for (let i = 0; i < cloth.nodes.length; i++) {
        const sphere = MeshBuilder.CreateSphere(`sphere${i}`, { diameter: sphereSize }, scene);
        sphere.position = cloth.nodes[i].position.clone();
        sphere.material = material;
        spheres.push(sphere);
    }

    scene.registerBeforeRender(() => {
        const deltaTime = engine.getDeltaTime() / 1000;

        solver.step(deltaTime);

        for (let i = 0; i < spheres.length; i++) {
            spheres[i].position.copyFrom(solver.positions[i]);
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