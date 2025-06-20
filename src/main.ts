import { Engine, Scene, ArcRotateCamera, Vector3, HemisphericLight, MeshBuilder } from "@babylonjs/core";

function createScene(engine: Engine, canvas: HTMLCanvasElement) : Scene {
  const scene = new Scene(engine);

  const camera = new ArcRotateCamera(
    "camera",
    Math.PI / 2,
    Math.PI / 4,
    10,
    Vector3.Zero(),
    scene
  );
  camera.attachControl(canvas, true);

  const light = new HemisphericLight(
    "light",
    new Vector3(0, 1, 0),
    scene
  );

  const sphere = MeshBuilder.CreateSphere(
    "sphere",
    { diameter: 2 },
    scene
  );

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