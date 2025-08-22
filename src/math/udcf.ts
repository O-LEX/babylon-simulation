import { Vector3 } from "@babylonjs/core";

export type Triangle = [Vector3, Vector3, Vector3];

function dot(a: Vector3, b: Vector3): number {
    return Vector3.Dot(a, b);
}

function cross(a: Vector3, b: Vector3): Vector3 {
    return Vector3.Cross(a, b);
}

function norm(a: Vector3): number {
    return a.length();
}

function normalize(v: Vector3): Vector3 {
    const n = v.length();
    if (n === 0) return v.clone();
    return v.scale(1 / n);
}

function swap(d0_: number, d1_: number, d2_: number, tri0: Vector3, tri1: Vector3, tri2: Vector3) {
    let v0_ = tri0;
    let v1_ = tri1;
    let v2_ = tri2;
    let dd0 = d0_;
    let dd1 = d1_;
    let dd2 = d2_;
    if ((dd0 <= 0 && dd1 >= 0 && dd2 >= 0) || (dd0 >= 0 && dd1 <= 0 && dd2 <= 0)) {
        v0_ = tri1;
        v1_ = tri0;
        v2_ = tri2;
        const d_ = dd0;
        dd0 = dd1;
        dd1 = d_;
        dd2 = dd2;
    } else if ((dd0 >= 0 && dd1 <= 0 && dd2 >= 0) || (dd0 <= 0 && dd1 >= 0 && dd2 <= 0)) {
        v0_ = tri0;
        v1_ = tri1;
        v2_ = tri2;
    } else if ((dd0 >= 0 && dd1 >= 0 && dd2 <= 0) || (dd0 <= 0 && dd1 <= 0 && dd2 >= 0)) {
        v0_ = tri0;
        v1_ = tri2;
        v2_ = tri1;
        const d_ = dd1;
        dd0 = dd0;
        dd1 = dd2;
        dd2 = d_;
    }
    return { d0_: dd0, d1_: dd1, d2_: dd2, v0_, v1_, v2_ };
}

function swap_minmax(t1_: number, t2_: number, d0_: number, d2_: number, v0_: Vector3, v2_: Vector3) {
    let t1 = t1_;
    let t2 = t2_;
    let dd0 = d0_;
    let dd2 = d2_;
    let vv0 = v0_;
    let vv2 = v2_;
    if (t1 > t2) {
        const t = t1;
        t1 = t2;
        t2 = t;
        const d = dd0;
        dd0 = dd2;
        dd2 = d;
        const v = vv0;
        vv0 = vv2;
        vv2 = v;
    }
    return { t1, t2, d0_: dd0, d2_: dd2, v0_: vv0, v2_: vv2 };
}

function gen_t(N_: Vector3, tri_: Triangle, D_: Vector3, d_: [number, number, number]): { t1: number, t2: number } | null {
    const e = 0.0;
    if (d_[0] <= e && d_[1] <= e && d_[2] <= e) return null;
    if (d_[0] >= -e && d_[1] >= -e && d_[2] >= -e) return null;
    const sw = swap(d_[0], d_[1], d_[2], tri_[0], tri_[1], tri_[2]);
    const p0_ = dot(D_, sw.v0_);
    const p1_ = dot(D_, sw.v1_);
    const p2_ = dot(D_, sw.v2_);
    let t1 = p0_ + (p1_ - p0_) * Math.abs(sw.d0_ / (sw.d0_ - sw.d1_));
    let t2 = p2_ + (p1_ - p2_) * Math.abs(sw.d2_ / (sw.d2_ - sw.d1_));
    const sm = swap_minmax(t1, t2, sw.d0_, sw.d2_, sw.v0_, sw.v2_);
    return { t1: sm.t1, t2: sm.t2 };
}

function line_intersection_on_same_plane(p1: Vector3, p2: Vector3, p3: Vector3, p4: Vector3, v1: Vector3, v2: Vector3): Vector3 | null {
    const d1 = p2.subtract(p1);
    const d2 = p4.subtract(p3);
    const n = cross(d1, d2);
    if (norm(n) === 0) return null;
    const denom = dot(n, n);
    if (denom === 0) return null;
    const v = p3.subtract(p1);
    const t1 = dot(cross(v, d2), n) / denom;
    const t2 = dot(cross(v, d1), n) / denom;
    if ((0 <= t2 && t2 <= 1) && (0 <= t1 && t1 <= 1)) {
        return v1.add(v2.subtract(v1).scale(t1));
    }
    return null;
}

function inside_triangle_on_same_plane(triangle: Triangle, p: Vector3): boolean {
    const ab = triangle[1].subtract(triangle[0]);
    const bp = p.subtract(triangle[1]);
    const bc = triangle[2].subtract(triangle[1]);
    const cp = p.subtract(triangle[2]);
    const ca = triangle[0].subtract(triangle[2]);
    const ap = p.subtract(triangle[0]);
    const c1 = cross(ab, bp);
    const c2 = cross(bc, cp);
    const c3 = cross(ca, ap);
    if (dot(c1, c2) > 0 && dot(c1, c3) > 0) return true;
    return false;
}

function find_intersection_point(N: Vector3, p0: Vector3, p: Vector3, line_dir: Vector3): Vector3 | null {
    if (dot(N, line_dir) === 0.0) return null;
    const t = dot(N, p0.subtract(p)) / dot(N, line_dir);
    const intersection_point = p.add(line_dir.scale(t));
    return intersection_point;
}

// Resolve triangle-triangle collision along the separating normal.
// Return value: the post-resolution vertex positions of the two triangles.
export function udcf(triangle1: Triangle, triangle2: Triangle): [Triangle, Triangle] {
    let N1 = cross(triangle1[1].subtract(triangle1[0]), triangle1[2].subtract(triangle1[0]));
    N1 = normalize(N1);
    const d1 = -dot(N1, triangle1[0]);
    let N2 = cross(triangle2[1].subtract(triangle2[0]), triangle2[2].subtract(triangle2[0]));
    N2 = normalize(N2);
    const d2 = -dot(N2, triangle2[0]);
    const d_on_vertex: [number[], number[]] = [[0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) d_on_vertex[0][i] = dot(N2, triangle1[i]) + d2;
    for (let i = 0; i < 3; i++) d_on_vertex[1][i] = dot(N1, triangle2[i]) + d1;
    let D = cross(N1, N2);
    const Dlen = norm(D);
    // Parallel planes or degenerate direction: no unique line of intersection -> no resolution
    if (Dlen === 0) return [
        [triangle1[0].clone(), triangle1[1].clone(), triangle1[2].clone()],
        [triangle2[0].clone(), triangle2[1].clone(), triangle2[2].clone()],
    ];
    D = D.scale(1 / Dlen);
    const gt1 = gen_t(N2, triangle1, D, [d_on_vertex[0][0], d_on_vertex[0][1], d_on_vertex[0][2]]);
    if (!gt1) return [
        [triangle1[0].clone(), triangle1[1].clone(), triangle1[2].clone()],
        [triangle2[0].clone(), triangle2[1].clone(), triangle2[2].clone()],
    ];
    const gt2 = gen_t(N1, triangle2, D, [d_on_vertex[1][0], d_on_vertex[1][1], d_on_vertex[1][2]]);
    if (!gt2) return [
        [triangle1[0].clone(), triangle1[1].clone(), triangle1[2].clone()],
        [triangle2[0].clone(), triangle2[1].clone(), triangle2[2].clone()],
    ];
    if (!(gt1.t2 >= gt2.t1 && gt2.t2 >= gt1.t1)) return [
        [triangle1[0].clone(), triangle1[1].clone(), triangle1[2].clone()],
        [triangle2[0].clone(), triangle2[1].clone(), triangle2[2].clone()],
    ];
    const P: [Triangle, Triangle] = [
        [
            triangle1[0].subtract(N2.scale(dot(N2, triangle1[0].subtract(triangle2[0])))),
            triangle1[1].subtract(N2.scale(dot(N2, triangle1[1].subtract(triangle2[0])))),
            triangle1[2].subtract(N2.scale(dot(N2, triangle1[2].subtract(triangle2[0])))),
        ],
        [
            triangle2[0].subtract(N1.scale(dot(N1, triangle2[0].subtract(triangle1[0])))),
            triangle2[1].subtract(N1.scale(dot(N1, triangle2[1].subtract(triangle1[0])))),
            triangle2[2].subtract(N1.scale(dot(N1, triangle2[2].subtract(triangle1[0])))),
        ],
    ];
    const Q0: Array<Vector3 | null> = [
        find_intersection_point(N1, triangle1[0], triangle2[0], N2),
        find_intersection_point(N1, triangle1[0], triangle2[1], N2),
        find_intersection_point(N1, triangle1[0], triangle2[2], N2),
    ];
    const Q1: Array<Vector3 | null> = [
        find_intersection_point(N2, triangle2[0], triangle1[0], N1),
        find_intersection_point(N2, triangle2[0], triangle1[1], N1),
        find_intersection_point(N2, triangle2[0], triangle1[2], N1),
    ];
    const intersections1: Vector3[] = [];
    let d_tri1 = 0.0;
    let d_tri2 = 0.0;
    for (let i = 0; i < 3; i++) {
        if (inside_triangle_on_same_plane(triangle2, P[0][i])) {
            if (d_tri1 < -d_on_vertex[0][i]) d_tri1 = -d_on_vertex[0][i];
        }
        if (Q0[i] !== null) {
            const q = Q0[i]!;
            if (inside_triangle_on_same_plane(triangle1, q)) intersections1.push(q);
        }
        for (let j = 0; j < 3; j++) {
            const inter = line_intersection_on_same_plane(
                P[0][i], P[0][(i + 1) % 3], triangle2[j], triangle2[(j + 1) % 3], triangle1[i], triangle1[(i + 1) % 3]
            );
            if (inter) intersections1.push(inter);
        }
    }
    const intersections2: Vector3[] = [];
    for (let i = 0; i < 3; i++) {
        if (inside_triangle_on_same_plane(triangle1, P[1][i])) {
            if (d_tri2 < -d_on_vertex[1][i]) d_tri2 = -d_on_vertex[1][i];
        }
        if (Q1[i] !== null) {
            const q = Q1[i]!;
            if (inside_triangle_on_same_plane(triangle2, q)) intersections2.push(q);
        }
        for (let j = 0; j < 3; j++) {
            const inter = line_intersection_on_same_plane(
                P[1][i], P[1][(i + 1) % 3], triangle1[j], triangle1[(j + 1) % 3], triangle2[i], triangle2[(i + 1) % 3]
            );
            if (inter) intersections2.push(inter);
        }
    }
    for (const v of intersections1) {
        const candidate = dot(N2, v) + d2;
        if (d_tri1 < -candidate) d_tri1 = -candidate;
    }
    for (const v of intersections2) {
        const candidate = dot(N1, v) + d1;
        if (d_tri2 < -candidate) d_tri2 = -candidate;
    }
    // Symmetric resolution: move both triangles by C/2 along a single separating normal.
    // Choose the side with smaller penetration (C), and split the correction.
    const eps = 1.0e-7;
    const pen1 = d_tri1 > eps ? d_tri1 : 0.0;
    const pen2 = d_tri2 > eps ? d_tri2 : 0.0;
    if (pen1 === 0.0 && pen2 === 0.0) {
        // No penetration
        return [
            [triangle1[0].clone(), triangle1[1].clone(), triangle1[2].clone()],
            [triangle2[0].clone(), triangle2[1].clone(), triangle2[2].clone()],
        ];
    }
    // Prefer the smaller positive penetration as C and use its corresponding normal.
    if (pen1 > 0.0 && (pen1 <= pen2 || pen2 === 0.0)) {
        const correction = N2.scale(pen1 * 0.5);
        const tri1Resolved: Triangle = [
            triangle1[0].add(correction),
            triangle1[1].add(correction),
            triangle1[2].add(correction),
        ];
        const tri2Resolved: Triangle = [
            triangle2[0].subtract(correction),
            triangle2[1].subtract(correction),
            triangle2[2].subtract(correction),
        ];
        return [tri1Resolved, tri2Resolved];
    }
    if (pen2 > 0.0) {
        const correction = N1.scale(pen2 * 0.5);
        const tri1Resolved: Triangle = [
            triangle1[0].subtract(correction),
            triangle1[1].subtract(correction),
            triangle1[2].subtract(correction),
        ];
        const tri2Resolved: Triangle = [
            triangle2[0].add(correction),
            triangle2[1].add(correction),
            triangle2[2].add(correction),
        ];
        return [tri1Resolved, tri2Resolved];
    }
    // Fallback (should not reach here given the checks above).
    return [
        [triangle1[0].clone(), triangle1[1].clone(), triangle1[2].clone()],
        [triangle2[0].clone(), triangle2[1].clone(), triangle2[2].clone()],
    ];
}

export function test(): [Triangle, Triangle] {
    const triangle1: Triangle = [
        new Vector3(0.375003, 0.299691, 0.299992),
        new Vector3(-0.224997, 0.299691, -0.300008),
        new Vector3(-0.224998, 0.299691, 0.299994),
    ];
    const triangle2: Triangle = [
        new Vector3(-0.0749846, 0.26, 0.0999057),
        new Vector3(0.125025, 0.26, 0.0999057),
        new Vector3(-0.1750912, 0.36939, 0.0999057),
    ];
    const [resolvedT1, resolvedT2] = udcf(triangle1, triangle2);
    console.log("resolvedT1:", resolvedT1);
    console.log("resolvedT2:", resolvedT2);
    return [resolvedT1, resolvedT2];
}
