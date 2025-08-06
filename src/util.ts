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

// Block sparse matrix using CSR format
export class BlockSparseMatrix {
    private row2idx: number[] = [];
    private idx2col: number[] = [];
    private idx2val: Matrix3x3[] = [];
    
    // Working arrays for CG solver
    private p: Vector3[] = [];
    private Ap: Vector3[] = [];

    initialize(row2col: number[], idx2col: number[]): void {
        this.row2idx = [...row2col];
        this.idx2col = [...idx2col];
        this.idx2val = new Array(idx2col.length).fill(null).map(() => new Matrix3x3());
        const n = row2col.length - 1;
        
        // Initialize working arrays
        this.p = new Array(n).fill(null).map(() => new Vector3(0, 0, 0));
        this.Ap = new Array(n).fill(null).map(() => new Vector3(0, 0, 0));
    }

    setZero(): void {
        for (let idx = 0; idx < this.idx2val.length; idx++) {
            this.idx2val[idx] = new Matrix3x3(); // Zero matrix
        }
    }

    addBlockAt(i_row: number, i_col: number, val: Matrix3x3): void {
        for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
            if (this.idx2col[idx] === i_col) {
                this.idx2val[idx] = this.idx2val[idx].add(val);
                return;
            }
        }
        console.error(`Block position (${i_row}, ${i_col}) not found in sparse matrix structure`);
    }

    setFixed(i_block: number): void {
        for (let j_block = 0; j_block < this.row2idx.length - 1; j_block++) {
            if (j_block === i_block) {
                // For the fixed block row
                for (let idx = this.row2idx[j_block]; idx < this.row2idx[j_block + 1]; idx++) {
                    if (this.idx2col[idx] === i_block) {
                        this.idx2val[idx] = this.idx2val[idx].add(Matrix3x3.identity());
                    } else {
                        this.idx2val[idx] = new Matrix3x3(); // Zero
                    }
                }
            } else {
                // For other rows, zero out the column corresponding to fixed block
                for (let idx = this.row2idx[j_block]; idx < this.row2idx[j_block + 1]; idx++) {
                    if (this.idx2col[idx] === i_block) {
                        this.idx2val[idx] = new Matrix3x3(); // Zero
                    }
                }
            }
        }
    }

    multiply(x: Vector3[], result: Vector3[]): void {
        for (let i_row = 0; i_row < this.row2idx.length - 1; i_row++) {
            result[i_row] = new Vector3(0, 0, 0);
            for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
                const j_col = this.idx2col[idx];
                result[i_row].addInPlace(this.idx2val[idx].multiplyVector(x[j_col]));
            }
        }
    }

    conjugateGradientSolver(r: Vector3[], maxIterations: number = 100, tolerance: number = 1e-6): Vector3[] {
        const n = r.length;
        const x = new Array(n).fill(null).map(() => new Vector3(0, 0, 0));
        
        // Copy r to p
        for (let i = 0; i < n; i++) {
            this.p[i] = r[i].clone();
        }
        
        let rsOld = 0;
        for (let i = 0; i < n; i++) {
            rsOld += Vector3.Dot(r[i], r[i]);
        }
        
        for (let iter = 0; iter < maxIterations; iter++) {
            this.multiply(this.p, this.Ap);
            
            let pAp = 0;
            for (let i = 0; i < n; i++) {
                pAp += Vector3.Dot(this.p[i], this.Ap[i]);
            }
            
            if (Math.abs(pAp) < 1e-12) {
                break;
            }
            
            const alpha = rsOld / pAp;
            
            for (let i = 0; i < n; i++) {
                x[i].addInPlace(this.p[i].scale(alpha));
                r[i].subtractInPlace(this.Ap[i].scale(alpha));
            }
            
            let rsNew = 0;
            for (let i = 0; i < n; i++) {
                rsNew += Vector3.Dot(r[i], r[i]);
            }
            
            if (Math.sqrt(rsNew) < tolerance) {
                break;
            }
            
            const beta = rsNew / rsOld;
            for (let i = 0; i < n; i++) {
                this.p[i] = r[i].add(this.p[i].scale(beta));
            }
            
            rsOld = rsNew;
        }
        
        return x;
    }
}

export class SparseMatrix {
    private row2idx: number[] = [];      // Row pointer
    private idx2col: number[] = [];      // Column indices
    private idx2val: number[] = [];      // Non-zero values

    // Working arrays for CG solver
    private p: number[] = [];
    private Ap: number[] = [];

    initialize(row2col: number[], idx2col: number[]): void {
        this.row2idx = [...row2col];
        this.idx2col = [...idx2col];
        this.idx2val = new Array(idx2col.length).fill(0);
        const n = row2col.length - 1;

        this.p = new Array(n).fill(0);
        this.Ap = new Array(n).fill(0);
    }

    setZero(): void {
        for (let i = 0; i < this.idx2val.length; i++) {
            this.idx2val[i] = 0;
        }
    }

    addValueAt(i_row: number, i_col: number, value: number): void {
        for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
            if (this.idx2col[idx] === i_col) {
                this.idx2val[idx] += value;
                return;
            }
        }
        console.error(`SparseMatrix: Position (${i_row}, ${i_col}) not found.`);
    }

    setFixed(i: number): void {
        for (let row = 0; row < this.row2idx.length - 1; row++) {
            for (let idx = this.row2idx[row]; idx < this.row2idx[row + 1]; idx++) {
                if (row === i && this.idx2col[idx] === i) {
                    this.idx2val[idx] += 1;
                } else if (row === i || this.idx2col[idx] === i) {
                    this.idx2val[idx] = 0;
                }
            }
        }
    }

    multiply(x: number[], result: number[]): void {
        for (let i = 0; i < this.row2idx.length - 1; i++) {
            result[i] = 0;
            for (let idx = this.row2idx[i]; idx < this.row2idx[i + 1]; idx++) {
                const j = this.idx2col[idx];
                result[i] += this.idx2val[idx] * x[j];
            }
        }
    }

    conjugateGradientSolver(r: number[], maxIterations = 100, tolerance = 1e-6): number[] {
        const n = r.length;
        const x = new Array(n).fill(0);
        this.p = [...r];

        let rsOld = 0;
        for (let i = 0; i < n; i++) {
            rsOld += r[i] * r[i];
        }

        for (let iter = 0; iter < maxIterations; iter++) {
            this.multiply(this.p, this.Ap);

            let pAp = 0;
            for (let i = 0; i < n; i++) {
                pAp += this.p[i] * this.Ap[i];
            }

            if (Math.abs(pAp) < 1e-12) break;

            const alpha = rsOld / pAp;

            for (let i = 0; i < n; i++) {
                x[i] += alpha * this.p[i];
                r[i] -= alpha * this.Ap[i];
            }

            let rsNew = 0;
            for (let i = 0; i < n; i++) {
                rsNew += r[i] * r[i];
            }

            if (Math.sqrt(rsNew) < tolerance) break;

            const beta = rsNew / rsOld;
            for (let i = 0; i < n; i++) {
                this.p[i] = r[i] + beta * this.p[i];
            }

            rsOld = rsNew;
        }

        return x;
    }
}
