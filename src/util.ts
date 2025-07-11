import { Vector3 } from "@babylonjs/core";

export class Matrix3x3 {
    data: number[] = new Array(9).fill(0);

    constructor(values?: number[]) {
        if (values) {
            this.data = [...values];
        }
    }

    static identity(): Matrix3x3 {
        const m = new Matrix3x3();
        m.data[0] = m.data[4] = m.data[8] = 1;
        return m;
    }

    static outerProduct(a: Vector3, b: Vector3): Matrix3x3 {
        const m = new Matrix3x3();
        m.data[0] = a.x * b.x; m.data[1] = a.x * b.y; m.data[2] = a.x * b.z;
        m.data[3] = a.y * b.x; m.data[4] = a.y * b.y; m.data[5] = a.y * b.z;
        m.data[6] = a.z * b.x; m.data[7] = a.z * b.y; m.data[8] = a.z * b.z;
        return m;
    }

    add(other: Matrix3x3): Matrix3x3 {
        const result = new Matrix3x3();
        for (let i = 0; i < 9; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    subtract(other: Matrix3x3): Matrix3x3 {
        const result = new Matrix3x3();
        for (let i = 0; i < 9; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    scale(s: number): Matrix3x3 {
        const result = new Matrix3x3();
        for (let i = 0; i < 9; i++) {
            result.data[i] = this.data[i] * s;
        }
        return result;
    }

    multiplyVector(v: Vector3): Vector3 {
        return new Vector3(
            this.data[0] * v.x + this.data[1] * v.y + this.data[2] * v.z,
            this.data[3] * v.x + this.data[4] * v.y + this.data[5] * v.z,
            this.data[6] * v.x + this.data[7] * v.y + this.data[8] * v.z
        );
    }

    solve(f: Vector3): Vector3 {
        const m = this.data;
        const a = m[0], b = m[1], c = m[2];
        const d = m[3], e = m[4], f_ = m[5];
        const g = m[6], h = m[7], i = m[8];

        const det = a * (e * i - f_ * h) - b * (d * i - f_ * g) + c * (d * h - e * g);
        if (Math.abs(det) < 1e-8) throw new Error("Matrix is singular");

        const invDet = 1 / det;
        const inv = new Matrix3x3([
            (e * i - f_ * h) * invDet,
            (c * h - b * i) * invDet,
            (b * f_ - c * e) * invDet,
            (f_ * g - d * i) * invDet,
            (a * i - c * g) * invDet,
            (c * d - a * f_) * invDet,
            (d * h - e * g) * invDet,
            (b * g - a * h) * invDet,
            (a * e - b * d) * invDet
        ]);
        return inv.multiplyVector(f);
    }
}