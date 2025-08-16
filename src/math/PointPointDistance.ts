import { Vector3 } from "@babylonjs/core";

export function val(d0: Vector3): number {
    return d0.lengthSquared();
}

export function grad(d0: Vector3): number[] {
    return [2 * d0.x, 2 * d0.y, 2 * d0.z];
}

export function hess(d0: Vector3): number[][] {
    const H: number[][] = Array(3).fill(0).map(() => Array(3).fill(0));
    H[0][0] = H[1][1] = H[2][2] = 2;
    return H;
}