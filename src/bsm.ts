// =================================================================================================
//
//  Single-File Implementation of Symmetric Block Sparse Matrix and Conjugate Gradient Solver
//
//  This file contains:
//  1. Matrix: A general-purpose class for dense matrices of arbitrary size.
//  2. VectorUtils: A namespace with helper functions for vector operations.
//  3. SymmetricBlockSparseMatrix: The main class for storing and operating on symmetric
//     block sparse matrices, storing only the upper-triangular blocks.
//  4. Test Case: A demonstration of how to create and use the symmetric matrix,
//     including a verification of the multiplication logic.
//
// =================================================================================================

/**
 * A general-purpose matrix class for arbitrary dimensions.
 */
export class Matrix {
    rows: number;
    cols: number;
    data: number[];

    constructor(rows: number, cols: number, values?: number[]) {
        this.rows = rows;
        this.cols = cols;
        this.data = new Array(rows * cols).fill(0);
        if (values && values.length === rows * cols) {
            this.data = [...values];
        }
    }

    /**
     * Creates an identity matrix of a given size.
     * @param size The width and height of the identity matrix.
     */
    static identity(size: number): Matrix {
        const m = new Matrix(size, size);
        for (let i = 0; i < size; i++) {
            m.data[i * size + i] = 1;
        }
        return m;
    }

    /**
     * Returns the transpose of the matrix.
     */
    transpose(): Matrix {
        const result = new Matrix(this.cols, this.rows);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[j * this.rows + i] = this.data[i * this.cols + j];
            }
        }
        return result;
    }
    
    /**
     * Multiplies the matrix by a vector (y = A * v).
     * @param v The vector, represented as a flat array.
     */
    multiplyVector(v: number[]): number[] {
        if (v.length !== this.cols) {
            throw new Error(`Matrix col count ${this.cols} does not match vector size ${v.length}`);
        }
        const result = new Array(this.rows).fill(0);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result[i] += this.data[i * this.cols + j] * v[j];
            }
        }
        return result;
    }
    
    /**
     * Multiplies the transpose of the matrix by a vector (y = A^T * v).
     * @param v The vector, represented as a flat array.
     */
    transposeMultiplyVector(v: number[]): number[] {
        if (v.length !== this.rows) {
            throw new Error(`Matrix row count ${this.rows} does not match vector size ${v.length} for transpose multiply`);
        }
        const result = new Array(this.cols).fill(0);
        for (let j = 0; j < this.cols; j++) {
            for (let i = 0; i < this.rows; i++) {
                result[j] += this.data[i * this.cols + j] * v[i];
            }
        }
        return result;
    }

    /**
     * Adds another matrix to this one.
     * @param other The matrix to add.
     */
    add(other: Matrix): Matrix {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Matrix dimensions must match for addition.");
        }
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    /**
     * Scales the matrix by a scalar value.
     * @param s The scalar value.
     */
    scale(s: number): Matrix {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.data.length; i++) {
            result.data[i] = this.data[i] * s;
        }
        return result;
    }
}

/**
 * A namespace containing helper functions for vector operations on flat arrays.
 */
export namespace VectorUtils {
    export function dot(a: number[], b: number[]): number {
        let result = 0;
        for (let i = 0; i < a.length; i++) result += a[i] * b[i];
        return result;
    }
    
    // Computes: r + p * beta
    export function add(r: number[], p: number[], beta: number): number[] {
        const result = new Array(r.length);
        for (let i = 0; i < r.length; i++) result[i] = r[i] + p[i] * beta;
        return result;
    }

    // Computes in-place: x += p * alpha
    export function addInPlace(x: number[], p: number[], alpha: number): void {
        for (let i = 0; i < x.length; i++) x[i] += p[i] * alpha;
    }

    // Computes in-place: r -= Ap * alpha
    export function subtractInPlace(r: number[], Ap: number[], alpha: number): void {
        for (let i = 0; i < r.length; i++) r[i] -= Ap[i] * alpha;
    }
}

/**
 * A symmetric block sparse matrix class using CSR format.
 * It only stores the upper-triangular blocks (where block j >= block i).
 */
export class SymmetricBlockSparseMatrix {
    // --- Structural Definition (Symmetric) ---
    private boundaries: number[] = [];      // Common block boundaries for rows and columns.
    private row2idx: number[] = [];         // CSR: row_ptr, points to the start of a row's data.
    private idx2col: number[] = [];         // CSR: indices array, contains column indices (only for j >= i).
    private idx2val: Matrix[] = [];         // CSR: data array, contains the upper-triangular Matrix blocks.

    // --- Workspace Arrays for CG Solver ---
    private p: number[] = [];
    private Ap: number[] = [];

    /**
     * Initializes the matrix structure.
     * @param bounds The array defining block boundaries (cumulative sizes).
     * @param row2idx The CSR row pointer array.
     * @param idx2col The CSR column indices array (must be upper-triangular).
     */
    initialize(
        bounds: number[],
        row2idx: number[],
        idx2col: number[]
    ): void {
        this.boundaries = [...bounds];
        this.row2idx = [...row2idx];
        this.idx2col = [...idx2col];
        this.idx2val = new Array(idx2col.length);

        // Pre-allocate Matrix objects with the correct dimensions based on the boundaries.
        for (let i_row = 0; i_row < this.numBlocks; i_row++) {
            const blockHeight = this.boundaries[i_row + 1] - this.boundaries[i_row];
            for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
                const j_col = this.idx2col[idx];
                if (j_col < i_row) {
                     throw new Error(`Symmetric matrix requires j_col >= i_row, but got (${i_row}, ${j_col}). Store upper-triangular blocks only.`);
                }
                const blockWidth = this.boundaries[j_col + 1] - this.boundaries[j_col];
                this.idx2val[idx] = new Matrix(blockHeight, blockWidth);
            }
        }
        
        const totalSize = bounds[bounds.length - 1];
        this.p = new Array(totalSize).fill(0);
        this.Ap = new Array(totalSize).fill(0);
    }
    
    get numBlocks(): number { return this.boundaries.length - 1; }

    /**
     * Adds a value to a specified block. Enforces upper-triangular storage.
     * @param i_row The row block index.
     * @param i_col The column block index.
     * @param val The Matrix object to add.
     */
    addBlockAt(i_row: number, i_col: number, val: Matrix): void {
        if (i_row > i_col) {
            throw new Error(`Cannot add block at (${i_row}, ${i_col}). Please add its transpose at (${i_col}, ${i_row}) instead.`);
        }
        for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
            if (this.idx2col[idx] === i_col) {
                this.idx2val[idx] = this.idx2val[idx].add(val);
                return;
            }
        }
        console.error(`Block position (${i_row}, ${i_col}) not found in sparse matrix structure`);
    }

    /**
     * Sets all matrix block values to zero.
     */
    setZero(): void {
        for (let i = 0; i < this.idx2val.length; i++) {
            const block = this.idx2val[i];
            if (block) { // Check if block exists
                for (let j = 0; j < block.data.length; j++) {
                    block.data[j] = 0;
                }
            }
        }
    }

    /**
     * Modifies the matrix to handle a fixed block (e.g., for boundary conditions).
     * Sets the diagonal block to identity and zeros out all other blocks in the
     * same row and column.
     * @param blockIndex The index of the block to fix.
     */
    setFixedBlock(blockIndex: number): void {
        const blockSize = this.boundaries[blockIndex + 1] - this.boundaries[blockIndex];

        // Zero out the row `blockIndex` (upper triangle part)
        for (let idx = this.row2idx[blockIndex]; idx < this.row2idx[blockIndex + 1]; idx++) {
            const j_col = this.idx2col[idx];
            const block = this.idx2val[idx];
            
            if (j_col === blockIndex) {
                // This is the diagonal block, set it to identity
                this.idx2val[idx] = Matrix.identity(blockSize);
            } else {
                // Off-diagonal block in the row, set to zero
                for (let k = 0; k < block.data.length; k++) block.data[k] = 0;
            }
        }

        // Zero out the column `blockIndex` (which corresponds to rows i < blockIndex)
        for (let i_row = 0; i_row < blockIndex; i_row++) {
            for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
                if (this.idx2col[idx] === blockIndex) {
                    const block = this.idx2val[idx];
                    for (let k = 0; k < block.data.length; k++) block.data[k] = 0;
                    break; // Found the block in this row, can move to the next row
                }
            }
        }
    }

    /**
     * Performs matrix-vector multiplication (y = A*x), leveraging symmetry.
     * @param x The input vector.
     * @param result The output vector (will be modified in-place).
     */
    multiply(x: number[], result: number[]): void {
        for (let i = 0; i < result.length; i++) result[i] = 0;
        
        // Iterate through the stored upper-triangular blocks.
        for (let i_row = 0; i_row < this.numBlocks; i_row++) {
            const rowStart = this.boundaries[i_row];
            const rowEnd = this.boundaries[i_row+1];
            const x_i_block = x.slice(rowStart, rowEnd);

            for (let idx = this.row2idx[i_row]; idx < this.row2idx[i_row + 1]; idx++) {
                const j_col = this.idx2col[idx];
                const blockMatrix = this.idx2val[idx];
                
                const colStart = this.boundaries[j_col];
                const colEnd = this.boundaries[j_col + 1];
                const x_j_block = x.slice(colStart, colEnd);

                // Add contribution from the upper-triangular part: y_i += A_ij * x_j
                const upperResult = blockMatrix.multiplyVector(x_j_block);
                for (let k = 0; k < upperResult.length; k++) {
                    result[rowStart + k] += upperResult[k];
                }

                // If not a diagonal block, add the contribution from the symmetric lower-triangular part.
                // y_j += A_ji * x_i  which is equal to  y_j += (A_ij)^T * x_i
                if (i_row !== j_col) {
                    const lowerResult = blockMatrix.transposeMultiplyVector(x_i_block);
                    for (let k = 0; k < lowerResult.length; k++) {
                        result[colStart + k] += lowerResult[k];
                    }
                }
            }
        }
    }
    
    /**
     * Solves the system Ax=b using the Conjugate Gradient method.
     * Assumes the matrix A is symmetric and positive-definite.
     * @param b The right-hand side vector.
     * @param maxIterations The maximum number of iterations.
     * @param tolerance The convergence tolerance.
     * @returns The solution vector x.
     */
    conjugateGradientSolver(b: number[], maxIterations: number = 100, tolerance: number = 1e-6): number[] {
        const n = b.length;
        const x = new Array(n).fill(0);
        const r = [...b]; // r = b - A*x (since x_0=0, r_0=b)
        this.p = [...r];

        let rsOld = VectorUtils.dot(r, r);
        
        for (let iter = 0; iter < maxIterations; iter++) {
            this.multiply(this.p, this.Ap);
            
            const pAp = VectorUtils.dot(this.p, this.Ap);
            
            if (Math.abs(pAp) < 1e-12) break;
            
            const alpha = rsOld / pAp;
            
            VectorUtils.addInPlace(x, this.p, alpha);
            VectorUtils.subtractInPlace(r, this.Ap, alpha);
            
            const rsNew = VectorUtils.dot(r, r);
            
            if (Math.sqrt(rsNew) < tolerance) break;
            
            this.p = VectorUtils.add(r, this.p, rsNew / rsOld);
            rsOld = rsNew;
        }
        
        return x;
    }
}


// =================================================================================================
//
//                                        TEST CASE
//
// =================================================================================================

console.log("=================================");
console.log("=== Symmetric Block Sparse Matrix Test ===");
console.log("=================================");

// 1. Define the block structure (mixing 2x2 and 3x3 blocks).
const bounds = [0, 2, 5]; // Block sizes: 2, 3 -> Total size 5x5
console.log("Block boundaries:", bounds);

// 2. Define the CSR structure (storing upper-triangular blocks only).
// Non-zero blocks will be at (0,0), (0,1), and (1,1).
const row2idx = [0, 2, 3];
const idx2col = [0, 1, 1];
console.log("CSR column indices (idx2col):", idx2col);

// 3. Initialize the matrix and set the block values.
const bsMatrix = new SymmetricBlockSparseMatrix();
bsMatrix.initialize(bounds, row2idx, idx2col);

// Block (0,0) - size 2x2
const m00 = new Matrix(2, 2, [10, 1, 1, 10]);
// Block (0,1) - size 2x3
const m01 = new Matrix(2, 3, [1, 2, 3, 4, 5, 6]);
// Block (1,1) - size 3x3
const m11 = new Matrix(3, 3, [20,0,0, 0,20,0, 0,0,20]);

bsMatrix.addBlockAt(0, 0, m00);
bsMatrix.addBlockAt(0, 1, m01);
bsMatrix.addBlockAt(1, 1, m11);

// Attempting to add to the lower triangle should fail.
try {
    bsMatrix.addBlockAt(1, 0, m01.transpose());
} catch(e: any) {
    console.log("\nSuccessfully caught an error when adding to the lower triangle.");
}

// 4. Test the matrix-vector multiplication.
const x_vec = [1, 2, 10, 11, 12]; // A vector of total size 5.
const result_vec = new Array(5).fill(0);
bsMatrix.multiply(x_vec, result_vec);

console.log("\nInput vector x:", x_vec);
console.log("Multiplication result A*x:", result_vec);

// 5. Verification.
// The full matrix A that is represented would be:
// [[10,  1,  1,  2,  3],
//  [ 1, 10,  4,  5,  6],
//  [ 1,  4, 20,  0,  0],  <- (A_01)^T
//  [ 2,  5,  0, 20,  0],
//  [ 3,  6,  0,  0, 20]]
//
// y = A * x
// y[0] = 10*1 + 1*2 + 1*10 + 2*11 + 3*12 = 10+2+10+22+36 = 80
// y[1] = 1*1 + 10*2 + 4*10 + 5*11 + 6*12 = 1+20+40+55+72 = 188
// y[2] = 1*1 + 4*2 + 20*10 = 1+8+200 = 209
// y[3] = 2*1 + 5*2 + 20*11 = 2+10+220 = 232
// y[4] = 3*1 + 6*2 + 20*12 = 3+12+240 = 255
console.log("Expected result:", [80, 188, 209, 232, 255]);