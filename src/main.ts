import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder, Mesh } from "@babylonjs/core";
import { SpringSim } from "./simulation";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
    const scene = new Scene(engine);

    const camera = new ArcRotateCamera("camera", Math.PI / 2, Math.PI / 4, 20, Vector3.Zero(), scene);
    camera.attachControl(canvas, true);

    const light = new HemisphericLight("light", new Vector3(0, 1, 0), scene);

    // const ground = MeshBuilder.CreateGround("ground", { width: 20, height: 20 }, scene);

    const fpsDisplay = document.getElementById("fpsDisplay");


    const pos: Vector3[] = [];
    for (let i = 0; i < 10; i++) {
        pos.push(new Vector3(i, 5, 0));
    }
    const mass: number[] = pos.map(() => 1);
    const stiffness = 100;
    const springSim = new SpringSim(pos, mass, stiffness);

    const sphereSize = 1.0;
    const spheres: Mesh[] = [];
    for (let i = 0; i < pos.length; i++) {
        const sphere = MeshBuilder.CreateSphere(`sphere${i}`, { diameter: sphereSize }, scene);
        sphere.position = pos[i].clone();
        spheres.push(sphere);
    }

    scene.registerBeforeRender(() => {
        const deltaTime = engine.getDeltaTime() / 1000;

        springSim.update(deltaTime);

        for (let i = 0; i < spheres.length; i++) {
            spheres[i].position.copyFrom(springSim.springPos[i]);
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