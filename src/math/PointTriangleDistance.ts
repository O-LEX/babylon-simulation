import { Vector3 } from "@babylonjs/core";
import * as PP from './PointPlaneDistance';
import * as PE from './PointEdgeDistance';

export function val(d0: Vector3, d1: Vector3, d2: Vector3): number {
    const n = Vector3.Cross(d1.subtract(d0), d2.subtract(d0));
    const s1 = Vector3.Dot(Vector3.Cross(d1, d2), n);
    const s2 = Vector3.Dot(Vector3.Cross(d2, d0), n);
    const s3 = Vector3.Dot(Vector3.Cross(d0, d1), n);
    
    if (s1 >= 0 && s2 >= 0 && s3 >= 0) {
        return PP.val(d0, d1, d2);
    }
    
    const dist01 = PE.val(d0, d1);
    const dist12 = PE.val(d1, d2);
    const dist20 = PE.val(d2, d0);
    
    return Math.min(dist01, dist12, dist20);
}

export function grad(d0: Vector3, d1: Vector3, d2: Vector3): number[] {
    const n = Vector3.Cross(d1.subtract(d0), d2.subtract(d0));
    const s1 = Vector3.Dot(Vector3.Cross(d1, d2), n);
    const s2 = Vector3.Dot(Vector3.Cross(d2, d0), n);
    const s3 = Vector3.Dot(Vector3.Cross(d0, d1), n);
    
    if (s1 >= 0 && s2 >= 0 && s3 >= 0) {
        return PP.grad(d0, d1, d2);
    }
    
    const dist01 = PE.val(d0, d1);
    const dist12 = PE.val(d1, d2);
    const dist20 = PE.val(d2, d0);
    
    if (dist01 <= dist12 && dist01 <= dist20) {
        return [...PE.grad(d0, d1), 0.0, 0.0, 0.0];
    }
    
    if (dist12 <= dist01 && dist12 <= dist20) {
        return [0.0, 0.0, 0.0, ...PE.grad(d1, d2)];
    }
    
    const grad = PE.grad(d2, d0);
    return [grad[3], grad[4], grad[5], 0.0, 0.0, 0.0, grad[0], grad[1], grad[2]];
}