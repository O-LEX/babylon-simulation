import { Vector3 } from "@babylonjs/core";

export function val(d0: Vector3, d1: Vector3): number {
    const e = d1.subtract(d0);
    const cross = Vector3.Cross(e, d0);
    const numerator = cross.lengthSquared();
    const denominator = e.lengthSquared();
    
    return numerator / denominator;
}

export function grad(d0: Vector3, d1: Vector3): number[] {
    const e = d1.subtract(d0);
    const eLengthSq = e.lengthSquared();

    const s = -Vector3.Dot(e, d0) / eLengthSq;
    
    // d = s*d0 + (1-s)*d1
    const d = d0.scale(s).add(d1.scale(1 - s));


    // dL/dd = 2d
    const dL_dd = d.scale(2);

    // dL/ds = (dL/dd) ⋅ (dd/ds)
    // dd/ds = d0 - d1 = -e
    const dd_ds = d0.subtract(d1);
    const dL_ds = Vector3.Dot(dL_dd, dd_ds);

    // dL/dd0_explicit = (dL/dd) ⋅ (dd/dd0) = (2d) ⋅ (s*I) = 2s*d
    const dL_dd0_explicit = dL_dd.scale(s);
    
    // dL/dd1_explicit = (dL/dd) ⋅ (dd/dd1) = (2d) ⋅ ((1-s)*I) = 2(1-s)d
    const dL_dd1_explicit = dL_dd.scale(1 - s);

    // ds/dd0 = (2*d0 - d1 - 2*s*e) / ||e||^2
    const ds_dd0_numerator = d0.scale(2).subtract(d1).subtract(e.scale(2 * s));
    const ds_dd0 = ds_dd0_numerator.scale(1 / eLengthSq);
    
    // ds/dd1 = (-d0 - 2*s*e) / ||e||^2
    const ds_dd1_numerator = d0.scale(-1).subtract(e.scale(2 * s));
    const ds_dd1 = ds_dd1_numerator.scale(1 / eLengthSq);
    
    // dL/dd0 = dL/dd0_explicit + (dL/ds) * (ds/dd0)
    const dL_dd0 = dL_dd0_explicit.add(ds_dd0.scale(dL_ds));
    
    // dL/dd1 = dL/dd1_explicit + (dL/ds) * (ds/dd1)
    const dL_dd1 = dL_dd1_explicit.add(ds_dd1.scale(dL_ds));
    
    const g: number[] = [
        dL_dd0.x,
        dL_dd0.y,
        dL_dd0.z,
        dL_dd1.x,
        dL_dd1.y,
        dL_dd1.z,
    ];

    return g;
}
