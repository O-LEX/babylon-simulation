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

    transpose(): Matrix3x3 {
        const m = new Matrix3x3();
        const d = this.data;
        m.data = [
            d[0], d[3], d[6],
            d[1], d[4], d[7],
            d[2], d[5], d[8]
        ];
        return m;
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
    numRows = 0;
    numCols = 0;
    row2idx: number[] = [];
    idx2col: number[] = [];
    idx2val: number[] = [];

    private p: number[] = [];
    private Ap: number[] = [];

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

export class CholeskySolver {
    private readonly n: number;
    private L: SparseMatrix; // The lower triangular Cholesky factor, stored internally.

    /**
     * Creates a Cholesky solver. The decomposition is computed upon instantiation.
     * @param A The symmetric positive-definite sparse matrix for the system Ax = b.
     * @throws {Error} If the matrix is not square or not positive-definite.
     */
    constructor(A: SparseMatrix) {
        if (A.numRows !== A.numCols) {
            throw new Error("Matrix must be square.");
        }
        this.n = A.numRows;

        // The Cholesky decomposition is performed once in the constructor.
        this.L = this.computeCholeskyFactor(A);
    }

    /**
     * Solves the linear system Ax = b for x.
     * @param b The right-hand side vector of the equation.
     * @returns The solution vector x.
     */
    public solve(b: Float32Array): Float32Array {
        if (b.length !== this.n) {
            throw new Error("Vector b has incorrect dimensions.");
        }

        // Step 1: Solve Ly = b using forward substitution.
        const y = this.forwardSubstitution(b);

        // Step 2: Solve L^T x = y using backward substitution.
        const x = this.backwardSubstitution(y);

        return x;
    }

    /**
     * Computes the Cholesky decomposition A = LL^T. This is a private helper method.
     * @param A The matrix to decompose.
     * @returns The lower triangular factor L as a SparseMatrix.
     */
    private computeCholeskyFactor(A: SparseMatrix): SparseMatrix {
        const n = A.numRows;
        // Store rows of L in maps for efficient sparse access during computation.
        const L_rows = Array.from({ length: n }, () => new Map<number, number>());

        // Helper function to get a value from the input matrix A.
        // This is slow for a large matrix, but simple. For production,
        // a binary search within the row's column indices would be faster.
        const getA = (row: number, col: number): number => {
            const rowStart = A.row2idx[row];
            const rowEnd = A.row2idx[row + 1];
            for (let i = rowStart; i < rowEnd; i++) {
                if (A.idx2col[i] === col) return A.idx2val[i];
                if (A.idx2col[i] > col) break; // Columns are sorted
            }
            return 0;
        };

        // Compute L column by column
        for (let j = 0; j < n; j++) {
            let s_diag = 0;
            const L_j_row = L_rows[j];

            // Calculate the dot product needed for the diagonal element L(j,j).
            // s_diag = Σ(k=0 to j-1) [L(j,k)]^2
            for (const [k, val] of L_j_row.entries()) {
                s_diag += val * val;
            }

            const Ajj = getA(j, j);
            const val_under_sqrt = Ajj - s_diag;

            // The matrix must be positive-definite, meaning this value must be positive.
            if (val_under_sqrt <= 1e-9) {
                throw new Error(`Matrix is not positive-definite. Failed at column ${j}.`);
            }
            const Ljj = Math.sqrt(val_under_sqrt);
            L_rows[j].set(j, Ljj);

            // Compute the off-diagonal elements in column j.
            for (let i = j + 1; i < n; i++) {
                let s_off_diag = 0;
                // s_off_diag = Σ(k=0 to j-1) [L(i,k) * L(j,k)]
                // This is a sparse dot product. Iterate over the shorter row for efficiency.
                const L_i_row = L_rows[i];
                const [iterMap, otherMap] = L_i_row.size < L_j_row.size ? [L_i_row, L_j_row] : [L_j_row, L_i_row];

                for (const [k, val1] of iterMap.entries()) {
                    if (k < j) {
                        const val2 = otherMap.get(k) || 0;
                        s_off_diag += val1 * val2;
                    }
                }

                const Aij = getA(i, j);
                const Lij = (Aij - s_off_diag) / Ljj;

                // To maintain sparsity, only store non-zero elements.
                if (Math.abs(Lij) > 1e-12) {
                    L_rows[i].set(j, Lij);
                }
            }
        }

        // Convert the array of maps into the final CSR format for the L matrix.
        const triplets: Triplet[] = [];
        for (let i = 0; i < n; i++) {
            for (const [j, val] of L_rows[i].entries()) {
                triplets.push({ row: i, col: j, val });
            }
        }

        const L = new SparseMatrix();
        L.resize(n, n);
        L.setFromTriplets(triplets);
        return L;
    }

    /**
     * Solves the lower-triangular system Ly = b for y.
     */
    private forwardSubstitution(b: Float32Array): Float32Array {
        const y = new Float32Array(this.n);
        for (let i = 0; i < this.n; i++) {
            let sum = 0;
            const rowStart = this.L.row2idx[i];
            const rowEnd = this.L.row2idx[i + 1];

            // sum = Σ L(i,j) * y(j) for j < i
            for (let idx = rowStart; idx < rowEnd; idx++) {
                const j = this.L.idx2col[idx];
                if (j < i) {
                    sum += this.L.idx2val[idx] * y[j];
                }
            }
            
            // For a lower-triangular matrix in CSR, the diagonal L(i,i)
            // is the last non-zero element in the row.
            const Lii = this.L.idx2val[rowEnd - 1];
            y[i] = (b[i] - sum) / Lii;
        }
        return y;
    }

    /**
     * Solves the upper-triangular system L^T x = y for x.
     */
    private backwardSubstitution(y: Float32Array): Float32Array {
        const x = new Float32Array(this.n);
        // Using an explicit transpose simplifies the implementation significantly.
        const LT = this.L.transpose();

        for (let i = this.n - 1; i >= 0; i--) {
            let sum = 0;
            const rowStart = LT.row2idx[i];
            const rowEnd = LT.row2idx[i + 1];

            // sum = Σ L^T(i,j) * x(j) for j > i
            for (let idx = rowStart; idx < rowEnd; idx++) {
                const j = LT.idx2col[idx];
                if (j > i) {
                    sum += LT.idx2val[idx] * x[j];
                }
            }

            // For an upper-triangular matrix (L^T), the diagonal L^T(i,i)
            // is the first non-zero element in the row.
            const LTii = LT.idx2val[rowStart];
            x[i] = (y[i] - sum) / LTii;
        }
        return x;
    }
}
