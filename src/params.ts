import { Vector3 } from "@babylonjs/core";

export interface Params {
    g: Vector3;
    dt: number;
    numSubsteps: number;
    numIterations: number;
}
