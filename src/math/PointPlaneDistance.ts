import { Vector3 } from "@babylonjs/core";

export function val(d0: Vector3, d1: Vector3, d2: Vector3): number {
    const e0 = d1.subtract(d0);
    const e1 = d2.subtract(d0);
    const n = Vector3.Cross(e0, e1);
    return (Vector3.Dot(d0, n) ** 2) / Vector3.Dot(n, n);
}

export function grad(d0: Vector3, d1: Vector3, d2: Vector3): number[] {
    const e0 = d1.subtract(d0);
    const e1 = d2.subtract(d0);
    const n = Vector3.Cross(e0, e1);
    
    const n2 = Vector3.Dot(n, n);
    const n_dot_d0 = Vector3.Dot(n, d0);
    
    const scalar_factor = (2 * n_dot_d0) / (n2 * n2);
    
    // Gradient with respect to d0
    const term1_d0 = Vector3.Cross(d1, d2).scale(n2);
    const term2_d0 = Vector3.Cross(d1.subtract(d2), n).scale(n_dot_d0);
    const grad_d0 = term1_d0.subtract(term2_d0).scale(scalar_factor);
    
    // Gradient with respect to d1
    const term1_d1 = Vector3.Cross(d2, d0).scale(n2);
    const term2_d1 = Vector3.Cross(d2.subtract(d0), n).scale(n_dot_d0);
    const grad_d1 = term1_d1.subtract(term2_d1).scale(scalar_factor);
    
    // Gradient with respect to d2
    const term1_d2 = Vector3.Cross(d0, d1).scale(n2);
    const term2_d2 = Vector3.Cross(d0.subtract(d1), n).scale(n_dot_d0);
    const grad_d2 = term1_d2.subtract(term2_d2).scale(scalar_factor);
    
    return [
        grad_d0.x, grad_d0.y, grad_d0.z,
        grad_d1.x, grad_d1.y, grad_d1.z,
        grad_d2.x, grad_d2.y, grad_d2.z
    ];
}