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

export interface Triplet {
    row: number;
    col: number;
    val: number;
}

export class SparseMatrix {
    private row2idx: number[] = [];
    private idx2col: number[] = [];
    private idx2val: number[] = [];

    private p: number[] = [];
    private Ap: number[] = [];

    private numRows = 0;
    private numCols = 0;

    // Set matrix size only (like Eigen::SparseMatrix::resize)
    resize(numRows: number, numCols: number): void {
        this.numRows = numRows;
        this.numCols = numCols;
        this.row2idx = [];
        this.idx2col = [];
        this.idx2val = [];
        this.p = new Array(numRows).fill(0);
        this.Ap = new Array(numRows).fill(0);
    }

    // Set structure and values from triplets (like Eigen::setFromTriplets)
    setFromTriplets(triplets: Triplet[]): void {
        const rowBuckets = Array.from({ length: this.numRows }, () => new Map<number, number>());

        for (const { row, col, val } of triplets) {
            if (row >= this.numRows || col >= this.numCols) {
                throw new Error(`Triplet out of bounds: (${row}, ${col})`);
            }

            const map = rowBuckets[row];
            if (map.has(col)) {
                map.set(col, map.get(col)! + val); // accumulate duplicates
            } else {
                map.set(col, val);
            }
        }

        this.row2idx = [0];
        this.idx2col = [];
        this.idx2val = [];

        for (const map of rowBuckets) {
            const sortedCols = Array.from(map.keys()).sort((a, b) => a - b);
            for (const col of sortedCols) {
                this.idx2col.push(col);
                this.idx2val.push(map.get(col)!);
            }
            this.row2idx.push(this.idx2col.length);
        }
    }

    multiplyVector(x: Float32Array): Float32Array {
        const ret = new Float32Array(this.numRows);
        const n = this.numRows;
        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let idx = this.row2idx[i]; idx < this.row2idx[i + 1]; idx++) {
                sum += this.idx2val[idx] * x[this.idx2col[idx]];
            }
            ret[i] = sum;
        }
        return ret;
    }

    transpose(): SparseMatrix {
        const triplets: Triplet[] = [];

        for (let row = 0; row < this.numRows; row++) {
            for (let idx = this.row2idx[row]; idx < this.row2idx[row + 1]; idx++) {
                const col = this.idx2col[idx];
                const val = this.idx2val[idx];
                triplets.push({ row: col, col: row, val }); // Flip row and column
            }
        }

        const transposed = new SparseMatrix();
        transposed.resize(this.numCols, this.numRows); // Swap rows and cols
        transposed.setFromTriplets(triplets);

        return transposed;
    }

    scale(scalar: number): SparseMatrix {
        const result = new SparseMatrix();
        result.resize(this.numRows, this.numCols);
        result.row2idx = [...this.row2idx];
        result.idx2col = [...this.idx2col];
        result.idx2val = this.idx2val.map(v => v * scalar);
        return result;
    }

    add(other: SparseMatrix): SparseMatrix {
        if (this.numRows !== other.numRows || this.numCols !== other.numCols) {
            throw new Error("Size mismatch in SparseMatrix.add()");
        }

        const triplets: Triplet[] = [];

        for (let row = 0; row < this.numRows; row++) {
            for (let i = this.row2idx[row]; i < this.row2idx[row + 1]; i++) {
                triplets.push({ row, col: this.idx2col[i], val: this.idx2val[i] });
            }
            for (let i = other.row2idx[row]; i < other.row2idx[row + 1]; i++) {
                triplets.push({ row, col: other.idx2col[i], val: other.idx2val[i] });
            }
        }

        const result = new SparseMatrix();
        result.resize(this.numRows, this.numCols);
        result.setFromTriplets(triplets);
        return result;
    }

    multiply(other: SparseMatrix): SparseMatrix {
        if (this.numCols !== other.numRows) {
            throw new Error("Size mismatch in SparseMatrix.multiply()");
        }

        // Convert 'other' to column-wise map for fast access
        const otherT = other.transpose(); // optional optimization
        const triplets: Triplet[] = [];

        for (let row = 0; row < this.numRows; row++) {
            const rowStart = this.row2idx[row];
            const rowEnd = this.row2idx[row + 1];

            for (let col = 0; col < other.numCols; col++) {
                let sum = 0;

                for (let i = rowStart; i < rowEnd; i++) {
                    const k = this.idx2col[i];
                    const Aik = this.idx2val[i];

                    const otherStart = other.row2idx[k];
                    const otherEnd = other.row2idx[k + 1];

                    for (let j = otherStart; j < otherEnd; j++) {
                        if (other.idx2col[j] === col) {
                            sum += Aik * other.idx2val[j];
                        }
                    }
                }

                if (Math.abs(sum) > 1e-12) {
                    triplets.push({ row, col, val: sum });
                }
            }
        }

        const result = new SparseMatrix();
        result.resize(this.numRows, other.numCols);
        result.setFromTriplets(triplets);
        return result;
    }

    conjugateGradientSolver(
        b: Float32Array,
        maxIterations: number = 100,
        tolerance: number = 1e-6
    ): Float32Array {
        const n = this.numRows;
        const x = new Float32Array(n); // initial guess: zero
        const r = new Float32Array(b); // residual r = b - A * x = b (initially)
        const p = new Float32Array(r); // search direction
        const Ap = new Float32Array(n);

        let rsOld = 0;
        for (let i = 0; i < n; i++) {
            rsOld += r[i] * r[i];
        }

        for (let iter = 0; iter < maxIterations; iter++) {
            const ApTemp = this.multiplyVector(p);
            for (let i = 0; i < n; i++) {
                Ap[i] = ApTemp[i];
            }

            let pAp = 0;
            for (let i = 0; i < n; i++) {
                pAp += p[i] * Ap[i];
            }

            if (Math.abs(pAp) < 1e-12) {
                break;
            }

            const alpha = rsOld / pAp;

            for (let i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            let rsNew = 0;
            for (let i = 0; i < n; i++) {
                rsNew += r[i] * r[i];
            }

            if (Math.sqrt(rsNew) < tolerance) {
                break;
            }

            const beta = rsNew / rsOld;

            for (let i = 0; i < n; i++) {
                p[i] = r[i] + beta * p[i];
            }

            rsOld = rsNew;
        }

        return x;
    }
}

export class Matrix {
  readonly rows: number;
  readonly cols: number;
  data: Float64Array;

  constructor(rows: number, cols: number, values?: number[] | Float64Array) {
    this.rows = rows;
    this.cols = cols;
    if (values) {
      if (values.length !== rows * cols) {
        throw new Error("Values length does not match matrix size");
      }
      this.data = new Float64Array(values);
    } else {
      this.data = new Float64Array(rows * cols);
    }
  }

  get(r: number, c: number): number {
    if (r < 0 || r >= this.rows || c < 0 || c >= this.cols) {
      throw new Error("Index out of bounds");
    }
    return this.data[r * this.cols + c];
  }

  set(r: number, c: number, val: number): void {
    if (r < 0 || r >= this.rows || c < 0 || c >= this.cols) {
      throw new Error("Index out of bounds");
    }
    this.data[r * this.cols + c] = val;
  }

  setZero(): void {
    this.data.fill(0);
  }

  static identity(size: number): Matrix {
    const m = new Matrix(size, size);
    for (let i = 0; i < size; i++) {
      m.set(i, i, 1);
    }
    return m;
  }

  add(other: Matrix): Matrix {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error("Matrix size mismatch in add");
    }
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i] + other.data[i];
    }
    return result;
  }

  scale(scalar: number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.data.length; i++) {
      result.data[i] = this.data[i] * scalar;
    }
    return result;
  }

  multiply(other: Matrix): Matrix {
    if (this.cols !== other.rows) {
      throw new Error("Matrix size mismatch in multiply");
    }
    const result = new Matrix(this.rows, other.cols);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < other.cols; c++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.get(r, k) * other.get(k, c);
        }
        result.set(r, c, sum);
      }
    }
    return result;
  }

  multiplyVector(x: number[]): number[] {
    if (x.length !== this.cols) throw new Error("vector length mismatch");
    const ret = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      let sum = 0;
      for (let j = 0; j < this.cols; j++) {
        sum += this.get(i, j) * x[j];
      }
      ret[i] = sum;
    }
    return ret;
  }

  transpose(): Matrix {
    const result = new Matrix(this.cols, this.rows);
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        result.set(c, r, this.get(r, c));
      }
    }
    return result;
  }
}
