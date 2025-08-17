import { Vector3 } from "@babylonjs/core";

export function val(d0: Vector3, d1: Vector3): number {
    const e = d1.subtract(d0);
    const n = Vector3.Cross(d0, d1);
    return Vector3.Dot(n, n) / Vector3.Dot(e, e);
}

export function grad(d0: Vector3, d1: Vector3): number[] {
    const e = d1.subtract(d0);
    const n = Vector3.Cross(d0, d1);
    const n2 = Vector3.Dot(n, n);
    const e2 = Vector3.Dot(e, e);
    
    const g_d0 = Vector3.Cross(d1, n).scale(e2).add(e.scale(n2)).scale(2 / (e2 * e2));
    const g_d1 = Vector3.Cross(d0, n).scale(e2).add(e.scale(n2)).scale(-2 / (e2 * e2));
    
    const g: number[] = [
        g_d0.x,
        g_d0.y,
        g_d0.z,
        g_d1.x,
        g_d1.y,
        g_d1.z,
    ];

    return g;
}
