import { Vector3 } from '@babylonjs/core';
import * as PP from './PointPointDistance';
import * as PL from './PointLineDistance';

export function val(d0: Vector3, d1: Vector3): number {
    const e = d1.subtract(d0);
    const s = - Vector3.Dot(e, d0) / Vector3.Dot(e, e);

    if (s < 0) {
        return PP.val(d0);
    } else if (s > 1) {
        return PP.val(d1);
    } else {
        return PL.val(d0, d1);
    }
}

export function grad(d0: Vector3, d1: Vector3): number[] {
    const e = d1.subtract(d0);
    const s = - Vector3.Dot(e, d0) / Vector3.Dot(e, e);

    if (s < 0) {    // point(p)-point(e0) expression
        const g_PP = PP.grad(d0);
        return [g_PP[0], g_PP[1], g_PP[2], 0.0, 0.0, 0.0];
    } else if (s > 1) {  // point(p)-point(e1) expression
        const g_PP = PP.grad(d1);
        return [0.0, 0.0, 0.0, g_PP[0], g_PP[1], g_PP[2]];
    } else {            // point(p)-line(e0e1) expression
        return PL.grad(d0, d1);
    }
}
