import { Vector3 } from "@babylonjs/core";

export function val(d0: Vector3): number {
    return d0.lengthSquared();
}

export function grad(d0: Vector3): number[] {
    return [2 * d0.x, 2 * d0.y, 2 * d0.z];
}
