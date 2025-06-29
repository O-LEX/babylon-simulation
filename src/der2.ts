import { Vector3 } from "@babylonjs/core";
import { SymmetricBlockSparseMatrix, Matrix } from "./bsm";

// --- Utility Functions for 3D Math ---

/**
 * Creates a 3x3 rotation matrix using Rodrigues' rotation formula.
 * @param axis The rotation axis (must be a unit vector).
 * @param angle The rotation angle in radians.
 * @returns A 3x3 rotation matrix.
 */
function createRotationMatrix(axis: Vector3, angle: number): Matrix {
    const cos_a = Math.cos(angle);
    const sin_a = Math.sin(angle);
    const one_minus_cos = 1.0 - cos_a;

    const x = axis.x;
    const y = axis.y;
    const z = axis.z;

    const data = [
        // Row 1
        cos_a + x * x * one_minus_cos,
        x * y * one_minus_cos - z * sin_a,
        x * z * one_minus_cos + y * sin_a,
        // Row 2
        y * x * one_minus_cos + z * sin_a,
        cos_a + y * y * one_minus_cos,
        y * z * one_minus_cos - x * sin_a,
        // Row 3
        z * x * one_minus_cos - y * sin_a,
        z * y * one_minus_cos + x * sin_a,
        cos_a + z * z * one_minus_cos
    ];

    return new Matrix(3, 3, data);
}

/**
 * Creates a 3x3 matrix from the outer product of two vectors.
 * @param a The first vector.
 * @param b The second vector.
 * @returns A 3x3 matrix.
 */
function createOuterProductMatrix(a: Vector3, b: Vector3): Matrix {
    const data = [
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z
    ];
    return new Matrix(3, 3, data);
}

function createSkewSymmetricMatrix(v: Vector3): Matrix {
    const data = [
        0, -v.z, v.y,
        v.z, 0, -v.x,
        -v.y, v.x, 0
    ];
    return new Matrix(3, 3, data);
}

/**
 * Multiplies a 3x3 Matrix by a Vector3.
 * @param matrix The 3x3 matrix to multiply.
 * @param v The vector to multiply.
 * @returns The transformed vector.
 */
function multiplyMatrixVector(matrix: Matrix, v: Vector3): Vector3 {
    if (matrix.rows !== 3 || matrix.cols !== 3) {
        throw new Error("Matrix must be 3x3 to multiply with a Vector3.");
    }
    const data = matrix.data;
    return new Vector3(
        data[0] * v.x + data[1] * v.y + data[2] * v.z,
        data[3] * v.x + data[4] * v.y + data[5] * v.z,
        data[6] * v.x + data[7] * v.y + data[8] * v.z
    );
}


export interface DERGeometry {
    positions: Vector3[]; // Positions of vertices
    fixedVertices: Set<number>; // Indices of fixed vertices
    stretchStiffness: number; // Stiffness for stretching
    bendStiffness: number; // Stiffness for bending
    twistStiffness: number; // Stiffness for twisting
    vertexMass: number; // Mass of each vertex
    radius: number; // Radius of the rod
}

// Discrete Elastic Rod solver using the new SymmetricBlockSparseMatrix
export class DERSolver {
    private q: number[] = []; // Flat array for all degrees of freedom (DOFs)
    private velocities: number[] = []; // Velocities for all DOFs
    
    private restLengths: number[] = [];
    private restTwists: number[] = [];
    private restCurvatures: Vector3[] = [];
    
    private fixedDofs: Set<number>; // Indices of fixed DOFs in the flat q array

    // Material properties
    private stretchStiffness: number;
    private bendStiffness: number;
    private twistStiffness: number;
    private vertexMass: number;
    private radius: number;

    private bsm: SymmetricBlockSparseMatrix;
    private gravity: Vector3 = new Vector3(0, 0, 0);

    // Frame data
    private tangents: Vector3[] = [];
    private ref_d1: Vector3[] = [];
    private ref_d2: Vector3[] = [];
    private mat_d1: Vector3[] = [];
    private mat_d2: Vector3[] = [];

    // Block boundaries for the sparse matrix
    private boundaries: number[] = [];
    private fixedBlockIndices: Set<number>;

    constructor(geometry: DERGeometry) {
        this.stretchStiffness = geometry.stretchStiffness;
        this.bendStiffness = geometry.bendStiffness;
        this.twistStiffness = geometry.twistStiffness;
        this.vertexMass = geometry.vertexMass;
        this.radius = geometry.radius;
        this.fixedDofs = new Set();
        this.fixedBlockIndices = new Set();

        // 1. Initialize q, boundaries, and fixedDofs
        this.initializeState(geometry);

        // 2. Compute rest state values
        this.computeRestState(geometry);

        // 3. Initialize frames
        this.updateTangents();
        this.updateReferenceFrames();
        this.updateMaterialFrames();

        // 4. Initialize the Symmetric Block Sparse Matrix
        this.bsm = new SymmetricBlockSparseMatrix();
        const structure = this.createSparseMatrixStructure();
        this.bsm.initialize(this.boundaries, structure.row2idx, structure.idx2col);
        
        // Initialize velocities to zero
        this.velocities = new Array(this.q.length).fill(0);
    }

    private initializeState(geometry: DERGeometry): void {
        const numVertices = geometry.positions.length;
        this.boundaries = [0];
        let currentBoundary = 0;

        for (let i = 0; i < numVertices; i++) {
            // Add position (3 DOFs)
            const pos = geometry.positions[i];
            this.q.push(pos.x, pos.y, pos.z);
            currentBoundary += 3;
            this.boundaries.push(currentBoundary);

            // Add theta (1 DOF)
            this.q.push(0); // Initial theta
            currentBoundary += 1;
            this.boundaries.push(currentBoundary);
        }

        // Initialize fixed DOFs and blocks
        for (const vertexIndex of geometry.fixedVertices) {
            const posBlockIndex = 2 * vertexIndex;
            this.fixedBlockIndices.add(posBlockIndex);
            const posDofStart = this.boundaries[posBlockIndex];
            this.fixedDofs.add(posDofStart);     // Fix x
            this.fixedDofs.add(posDofStart + 1); // Fix y
            this.fixedDofs.add(posDofStart + 2); // Fix z

            const thetaBlockIndex = 2 * vertexIndex + 1;
            this.fixedBlockIndices.add(thetaBlockIndex);
            const thetaDofStart = this.boundaries[thetaBlockIndex];
            this.fixedDofs.add(thetaDofStart); // Fix theta
        }
    }

    private computeRestState(geometry: DERGeometry): void {
        const numEdges = geometry.positions.length - 1;
        this.restLengths = new Array(numEdges);
        this.restTwists = new Array(numEdges).fill(0);
        this.restCurvatures = new Array(numEdges).fill(new Vector3(0,0,0));

        for (let i = 0; i < numEdges; i++) {
            this.restLengths[i] = Vector3.Distance(geometry.positions[i], geometry.positions[i + 1]);
        }
    }

    private createSparseMatrixStructure(): { row2idx: number[], idx2col: number[] } {
        const numBlocks = this.boundaries.length - 1;
        const adjacency = new Map<number, Set<number>>();

        // Initialize adjacency with self-connections (diagonal)
        for (let i = 0; i < numBlocks; i++) {
            adjacency.set(i, new Set([i]));
        }

        const numVertices = (numBlocks) / 2;

        // Connect blocks based on physical interactions
        for (let i = 0; i < numVertices - 1; i++) {
            const p_i = 2 * i;       // Position block for vertex i
            const t_i = 2 * i + 1;   // Theta block for vertex i
            const p_i1 = 2 * (i + 1); // Position block for vertex i+1
            const t_i1 = 2 * (i + 1) + 1; // Theta block for vertex i+1

            // Stretching connects p_i and p_i1
            adjacency.get(p_i)!.add(p_i1);

            // Twisting connects t_i and t_i1
            adjacency.get(t_i)!.add(t_i1);

            // Bending involves three consecutive vertices, so connect p_i, p_i1, and p_i2
            if (i < numVertices - 2) {
                const p_i2 = 2 * (i + 2);
                adjacency.get(p_i)!.add(p_i1).add(p_i2);
                adjacency.get(p_i1)!.add(p_i2);
            }
        }

        // Build CSR structure (upper-triangular)
        const row2idx: number[] = [0];
        const idx2col: number[] = [];

        for (let i = 0; i < numBlocks; i++) {
            // Sort neighbors to ensure upper-triangular storage (j >= i)
            const neighbors = Array.from(adjacency.get(i)!).filter(j => j >= i).sort((a, b) => a - b);
            for (const neighbor of neighbors) {
                idx2col.push(neighbor);
            }
            row2idx.push(idx2col.length);
        }

        return { row2idx, idx2col };
    }

    // --- Helper Functions to access state ---

    private getNumVertices(): number {
        return (this.boundaries.length - 1) / 2;
    }

    private getNumEdges(): number {
        return this.getNumVertices() - 1;
    }

    private getPosition(vertexIndex: number): Vector3 {
        const blockIndex = 2 * vertexIndex;
        const dofIndex = this.boundaries[blockIndex];
        return new Vector3(this.q[dofIndex], this.q[dofIndex + 1], this.q[dofIndex + 2]);
    }

    private getTheta(vertexIndex: number): number {
        const blockIndex = 2 * vertexIndex + 1;
        const dofIndex = this.boundaries[blockIndex];
        return this.q[dofIndex];
    }

    // --- Frame Computation ---

    /**
     * Updates the tangent vectors for each edge based on current positions.
     */
    private updateTangents() {
        this.tangents = [];
        for (let i = 0; i < this.getNumEdges(); i++) {
            const p0 = this.getPosition(i);
            const p1 = this.getPosition(i + 1);
            this.tangents.push(p1.subtract(p0).normalize());
        }
    }

    /**
     * Transports a vector along the rod without twisting.
     */
    private parallelTransport(vectorToTransport: Vector3, t_prev: Vector3, t_curr: Vector3): Vector3 {
        const axis = Vector3.Cross(t_prev, t_curr);
        const axis_len = axis.length();

        if (axis_len < 1e-8) {
            return vectorToTransport.clone(); // No rotation needed
        }

        const angle = Math.acos(Vector3.Dot(t_prev, t_curr));
        const R = createRotationMatrix(axis.normalize(), angle);
        return multiplyMatrixVector(R, vectorToTransport);
    }

    /**
     * Updates the reference frames (d1, d2) along the rod.
     */
    private updateReferenceFrames() {
        this.ref_d1 = [];
        this.ref_d2 = [];

        for (let i = 0; i < this.getNumEdges(); i++) {
            const t = this.tangents[i];
            let d1: Vector3;

            if (i === 0) {
                // For the first edge, create an arbitrary frame
                const up = Math.abs(Vector3.Dot(t, Vector3.Up())) > 0.9 ? Vector3.Right() : Vector3.Up();
                d1 = Vector3.Cross(t, up).normalize();
            } else {
                // For subsequent edges, parallel transport the previous frame
                const d1_prev = this.ref_d1[i - 1];
                const t_prev = this.tangents[i - 1];
                d1 = this.parallelTransport(d1_prev, t_prev, t);
            }
            const d2 = Vector3.Cross(t, d1).normalize();
            this.ref_d1.push(d1);
            this.ref_d2.push(d2);
        }
    }

    /**
     * Updates the material frames based on the reference frames and twist angles (theta).
     */
    private updateMaterialFrames() {
        this.mat_d1 = [];
        this.mat_d2 = [];
        for (let i = 0; i < this.getNumEdges(); i++) {
            const t = this.tangents[i];
            const theta = this.getTheta(i); // Twist at the start of the edge
            const d1_ref = this.ref_d1[i];

            // Rotate the reference frame by the twist angle
            const R_twist = createRotationMatrix(t, theta);
            const d1_mat = multiplyMatrixVector(R_twist, d1_ref);
            const d2_mat = Vector3.Cross(t, d1_mat);

            this.mat_d1.push(d1_mat);
            this.mat_d2.push(d2_mat);
        }
    }

    // --- Force Computations ---

    /**
     * Computes bending forces, gradients, and Hessians for a single vertex.
     * Based on the Discrete Elastic Rods model.
     * @param vertexIndex The index of the central vertex (x_i) for the bend.
     * @param stiffness The bending stiffness coefficient.
     */
    private computeBendingForces(
        vertexIndex: number,
        stiffness: number
    ): {
        energy: number;
        gradient: number[];
        hessian: Matrix[][];
    } {
        // --- 1. Initial Setup and Geometric Quantities ---
        const x_prev = this.getPosition(vertexIndex - 1);
        const x_curr = this.getPosition(vertexIndex);
        const x_next = this.getPosition(vertexIndex + 1);

        const e_prev = x_curr.subtract(x_prev);
        const e_curr = x_next.subtract(x_curr);

        const l_prev = e_prev.length();
        const l_curr = e_curr.length();

        const zeros = {
            energy: 0,
            gradient: Array(9).fill(0),
            hessian: Array(3).fill(null).map(() => Array(3).fill(new Matrix(3, 3)))
        };

        if (l_prev < 1e-9 || l_curr < 1e-9) return zeros;

        const t_prev = e_prev.scale(1.0 / l_prev);
        const t_curr = e_curr.scale(1.0 / l_curr);

        const chi = 1.0 + Vector3.Dot(t_prev, t_curr);
        if (Math.abs(chi) < 1e-9) return zeros;

        const kappa_b = Vector3.Cross(t_prev, t_curr).scale(2.0 / chi); // [cite: 487]

        // Material frames and curvature calculation
        const mat_d1_prev = this.mat_d1[vertexIndex - 1];
        const mat_d2_prev = this.mat_d2[vertexIndex - 1];
        const mat_d1_curr = this.mat_d1[vertexIndex];
        const mat_d2_curr = this.mat_d2[vertexIndex];
        
        const kappa1 = 0.5 * Vector3.Dot(mat_d2_prev.add(mat_d2_curr), kappa_b); // [cite: 489]
        const kappa2 = -0.5 * Vector3.Dot(mat_d1_prev.add(mat_d1_curr), kappa_b); // [cite: 489]

        const kappa1_rest = 0.0; // Assuming zero rest curvature
        const kappa2_rest = 0.0;
        
        const kappa1_diff = kappa1 - kappa1_rest;
        const kappa2_diff = kappa2 - kappa2_rest;
        
        const voronoi_length = 0.5 * (this.restLengths[vertexIndex - 1] + this.restLengths[vertexIndex]);
        if (voronoi_length < 1e-9) return zeros;

        const energy_factor = stiffness / voronoi_length;
        const energy = 0.5 * energy_factor * (kappa1_diff * kappa1_diff + kappa2_diff * kappa2_diff);

        // Coefficients needed for energy gradient calculation
        const dE_dKappa1 = energy_factor * kappa1_diff;
        const dE_dKappa2 = energy_factor * kappa2_diff;

        // --- 2. First Derivative of Curvature (Gradient) Calculation ---
        // Calculate tilde vectors as per the paper's definition [cite: 489]
        const t_tilde = t_prev.add(t_curr).scale(1.0 / chi);
        const d1_tilde = mat_d1_prev.add(mat_d1_curr);
        const d2_tilde = mat_d2_prev.add(mat_d2_curr);

        // ∂κ₁/∂e and ∂κ₂/∂e [cite: 495]
        const dK1de_prev = t_tilde.scale(-kappa1).add(Vector3.Cross(t_curr, d2_tilde.scale(0.5))).scale(1.0 / l_prev);
        const dK1de_curr = t_tilde.scale(-kappa1).subtract(Vector3.Cross(t_prev, d2_tilde.scale(0.5))).scale(1.0 / l_curr);
        const dK2de_prev = t_tilde.scale(-kappa2).subtract(Vector3.Cross(t_curr, d1_tilde.scale(0.5))).scale(1.0 / l_prev);
        const dK2de_curr = t_tilde.scale(-kappa2).add(Vector3.Cross(t_prev, d1_tilde.scale(0.5))).scale(1.0 / l_curr);

        // Use chain rule to compute energy gradient ∂E/∂x
        const grad_e_prev = dK1de_prev.scale(dE_dKappa1).add(dK2de_prev.scale(dE_dKappa2));
        const grad_e_curr = dK1de_curr.scale(dE_dKappa1).add(dK2de_curr.scale(dE_dKappa2));

        const grad_x_prev = grad_e_prev.scale(-1);
        const grad_x_next = grad_e_curr;
        const grad_x_curr = grad_e_prev.subtract(grad_e_curr);

        const gradient = [
            grad_x_prev.x, grad_x_prev.y, grad_x_prev.z,
            grad_x_curr.x, grad_x_curr.y, grad_x_curr.z,
            grad_x_next.x, grad_x_next.y, grad_x_next.z
        ];

        // --- 3. Hessian of Curvature Calculation ---
        const I3 = Matrix.identity(3);

        // Term 1 of Hessian (Gauss-Newton term)
        const grad_k1_vectors = [dK1de_prev.scale(-1), dK1de_prev.add(dK1de_curr), dK1de_curr.scale(-1)];
        const grad_k2_vectors = [dK2de_prev.scale(-1), dK2de_prev.add(dK2de_curr), dK2de_curr.scale(-1)];
        const H_GN: Matrix[][] = Array(3).fill(null).map(() => Array(3).fill(new Matrix(3,3)));
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                const H_GN_k1 = createOuterProductMatrix(grad_k1_vectors[i], grad_k1_vectors[j]);
                const H_GN_k2 = createOuterProductMatrix(grad_k2_vectors[i], grad_k2_vectors[j]);
                H_GN[i][j] = H_GN_k1.add(H_GN_k2).scale(energy_factor);
            }
        }
        
        // Term 2 of Hessian (second derivative of curvature term)
        const H_k_deriv: Matrix[][] = Array(3).fill(null).map(() => Array(3).fill(new Matrix(3,3)));
        
        // Calculate ∂²κ₁/∂e² and ∂²κ₂/∂e² [cite: 500-503]
        // 3.1 Calculate H_k1
        // ∂²κ₁/∂(e^{i-1})²
        const t_tilde_outer = createOuterProductMatrix(t_tilde, t_tilde);
        const H1_pp_term1 = t_tilde_outer.scale(2 * kappa1)
            .subtract(createOuterProductMatrix(Vector3.Cross(t_curr, d2_tilde), t_tilde))
            .subtract(createOuterProductMatrix(t_tilde, Vector3.Cross(t_curr, d2_tilde)))
            .scale(1.0 / (l_prev * l_prev));
        const H1_pp_term2 = I3.subtract(createOuterProductMatrix(t_prev, t_prev)).scale(-kappa1 / (chi * l_prev * l_prev));
        const H1_pp_term3 = createOuterProductMatrix(kappa_b, this.mat_d2[vertexIndex-1]).add(createOuterProductMatrix(this.mat_d2[vertexIndex-1], kappa_b)).scale(1.0 / (4 * l_prev * l_prev));
        const H1_pp = H1_pp_term1.add(H1_pp_term2).add(H1_pp_term3);

        // ∂²κ₁/∂(e^i)²
        const H1_nn_term1 = t_tilde_outer.scale(2 * kappa1)
            .add(createOuterProductMatrix(Vector3.Cross(t_prev, d2_tilde), t_tilde))
            .add(createOuterProductMatrix(t_tilde, Vector3.Cross(t_prev, d2_tilde)))
            .scale(1.0 / (l_curr * l_curr));
        const H1_nn_term2 = I3.subtract(createOuterProductMatrix(t_curr, t_curr)).scale(-kappa1 / (chi * l_curr * l_curr));
        const H1_nn_term3 = createOuterProductMatrix(kappa_b, this.mat_d2[vertexIndex]).add(createOuterProductMatrix(this.mat_d2[vertexIndex], kappa_b)).scale(1.0 / (4 * l_curr * l_curr));
        const H1_nn = H1_nn_term1.add(H1_nn_term2).add(H1_nn_term3);

        // ∂²κ₁/∂e^{i-1}∂e^i
        const H1_pn_term1 = I3.add(createOuterProductMatrix(t_prev, t_curr)).scale(-kappa1 / (chi * l_prev * l_curr));
        const H1_pn_term2 = t_tilde_outer.scale(2 * kappa1)
            .subtract(createOuterProductMatrix(Vector3.Cross(t_curr, d2_tilde), t_tilde))
            .add(createOuterProductMatrix(t_tilde, Vector3.Cross(t_prev, d2_tilde)))
            .subtract(createSkewSymmetricMatrix(d2_tilde))
            .scale(1.0 / (l_prev * l_curr));
        const H1_pn = H1_pn_term1.add(H1_pn_term2);
        
        // 3.2 Calculate H_k2 (κ₁ -> κ₂, d₂ -> -d₁)
        // ∂²κ₂/∂(e^{i-1})²
        const H2_pp_term1 = t_tilde_outer.scale(2 * kappa2)
            .subtract(createOuterProductMatrix(Vector3.Cross(t_curr, d1_tilde).scale(-1), t_tilde))
            .subtract(createOuterProductMatrix(t_tilde, Vector3.Cross(t_curr, d1_tilde).scale(-1)))
            .scale(1.0 / (l_prev * l_prev));
        const H2_pp_term2 = I3.subtract(createOuterProductMatrix(t_prev, t_prev)).scale(-kappa2 / (chi * l_prev * l_prev));
        const H2_pp_term3 = createOuterProductMatrix(kappa_b, this.mat_d1[vertexIndex-1]).add(createOuterProductMatrix(this.mat_d1[vertexIndex-1], kappa_b)).scale(-1.0 / (4 * l_prev * l_prev));
        const H2_pp = H2_pp_term1.add(H2_pp_term2).add(H2_pp_term3);
        
        // ∂²κ₂/∂(e^i)²
        const H2_nn_term1 = t_tilde_outer.scale(2 * kappa2)
            .add(createOuterProductMatrix(Vector3.Cross(t_prev, d1_tilde).scale(-1), t_tilde))
            .add(createOuterProductMatrix(t_tilde, Vector3.Cross(t_prev, d1_tilde).scale(-1)))
            .scale(1.0 / (l_curr * l_curr));
        const H2_nn_term2 = I3.subtract(createOuterProductMatrix(t_curr, t_curr)).scale(-kappa2 / (chi * l_curr * l_curr));
        const H2_nn_term3 = createOuterProductMatrix(kappa_b, this.mat_d1[vertexIndex]).add(createOuterProductMatrix(this.mat_d1[vertexIndex], kappa_b)).scale(-1.0 / (4 * l_curr * l_curr));
        const H2_nn = H2_nn_term1.add(H2_nn_term2).add(H2_nn_term3);

        // ∂²κ₂/∂e^{i-1}∂e^i
        const H2_pn_term1 = I3.add(createOuterProductMatrix(t_prev, t_curr)).scale(-kappa2 / (chi * l_prev * l_curr));
        const H2_pn_term2 = t_tilde_outer.scale(2 * kappa2)
            .subtract(createOuterProductMatrix(Vector3.Cross(t_curr, d1_tilde).scale(-1), t_tilde))
            .add(createOuterProductMatrix(t_tilde, Vector3.Cross(t_prev, d1_tilde).scale(-1)))
            .subtract(createSkewSymmetricMatrix(d1_tilde).scale(-1))
            .scale(1.0 / (l_prev * l_curr));
        const H2_pn = H2_pn_term1.add(H2_pn_term2);

        // 3.3 Apply chain rule to calculate ∂²E/∂x²
        // ∂²E/∂e_a ∂e_b = dE/dκ₁ * ∂²κ₁/∂e_a ∂e_b + dE/dκ₂ * ∂²κ₂/∂e_a ∂e_b
        const H_deriv_pp = H1_pp.scale(dE_dKappa1).add(H2_pp.scale(dE_dKappa2));
        const H_deriv_nn = H1_nn.scale(dE_dKappa1).add(H2_nn.scale(dE_dKappa2));
        const H_deriv_pn = H1_pn.scale(dE_dKappa1).add(H2_pn.scale(dE_dKappa2));
        const H_deriv_np = H_deriv_pn.transpose();

        // Convert to Hessian for vertex coordinates x
        H_k_deriv[0][0] = H_deriv_pp;
        H_k_deriv[0][1] = H_deriv_pp.add(H_deriv_np).scale(-1);
        H_k_deriv[0][2] = H_deriv_np;
        H_k_deriv[1][0] = H_k_deriv[0][1].transpose();
        H_k_deriv[1][1] = H_deriv_pp.add(H_deriv_nn).add(H_deriv_pn).add(H_deriv_np);
        H_k_deriv[1][2] = H_deriv_nn.add(H_deriv_pn).scale(-1);
        H_k_deriv[2][0] = H_k_deriv[0][2].transpose();
        H_k_deriv[2][1] = H_k_deriv[1][2].transpose();
        H_k_deriv[2][2] = H_deriv_nn;

        // --- 4. Assemble Final Hessian ---
        const hessian: Matrix[][] = Array(3).fill(null).map(() => Array(3).fill(new Matrix(3,3)));
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                // Sum of Gauss-Newton term and second derivative term
                hessian[i][j] = H_GN[i][j].add(H_k_deriv[i][j]);
            }
        }

        return { energy, gradient, hessian };
    }

    // Computes stretching energy, gradient, and Hessian blocks.
    private computeStretchingForces(
        p0_idx: number, // start index of p0 in q
        p1_idx: number, // start index of p1 in q
        restLength: number,
        stiffness: number
    ): { 
        energy: number, 
        gradient: number[], 
        hessian: { H_p0p0: Matrix, H_p0p1: Matrix, H_p1p1: Matrix } 
    } {
        const p0 = new Vector3(this.q[p0_idx], this.q[p0_idx + 1], this.q[p0_idx + 2]);
        const p1 = new Vector3(this.q[p1_idx], this.q[p1_idx + 1], this.q[p1_idx + 2]);

        const edge = p1.subtract(p0);
        const currentLength = edge.length();

        if (currentLength < 1e-8) {
            const zeroHessian = new Matrix(3, 3);
            return {
                energy: 0,
                gradient: [0, 0, 0, 0, 0, 0],
                hessian: {
                    H_p0p0: zeroHessian,
                    H_p0p1: zeroHessian,
                    H_p1p1: zeroHessian
                }
            };
        }

        const strain = currentLength - restLength;
        const energy = 0.5 * stiffness * strain * strain;

        const u = edge.normalize();

        // Gradient is a 6x1 vector [gx0, gy0, gz0, gx1, gy1, gz1]
        const grad_p0 = u.scale(-stiffness * strain);
        const grad_p1 = u.scale(stiffness * strain);
        const gradient = [grad_p0.x, grad_p0.y, grad_p0.z, grad_p1.x, grad_p1.y, grad_p1.z];

        // Hessian are 3x3 matrices
        const u_outer_u = [
            u.x * u.x, u.x * u.y, u.x * u.z,
            u.y * u.x, u.y * u.y, u.y * u.z,
            u.z * u.x, u.z * u.y, u.z * u.z
        ];

        const H_block = new Matrix(3, 3, u_outer_u.map(v => v * stiffness));
        const H_block_neg = H_block.scale(-1);

        return {
            energy,
            gradient,
            hessian: {
                H_p0p0: H_block,
                H_p0p1: H_block_neg,
                H_p1p1: H_block
            }
        };
    }

    public step(deltaTime: number): void {
        const numDofs = this.q.length;
        const gradient = new Array(numDofs).fill(0);
        
        this.bsm.setZero();

        // Step 1: Update q using current velocities (prediction)
        for (let i = 0; i < numDofs; i++) {
            if (!this.fixedDofs.has(i)) {
                this.q[i] += this.velocities[i] * deltaTime;
            }
        }

        // Update geometric frames
        this.updateTangents();
        this.updateReferenceFrames();
        this.updateMaterialFrames();

        // Step 2: Process stretching forces
        for (let i = 0; i < this.getNumEdges(); i++) {
            const v0_idx = i;
            const v1_idx = i + 1;

            const p0_block_idx = 2 * v0_idx;
            const p1_block_idx = 2 * v1_idx;

            const p0_dof_start = this.boundaries[p0_block_idx];
            const p1_dof_start = this.boundaries[p1_block_idx];

            const result = this.computeStretchingForces(
                p0_dof_start,
                p1_dof_start,
                this.restLengths[i],
                this.stretchStiffness
            );

            // Assemble gradient
            for (let d = 0; d < 3; d++) {
                gradient[p0_dof_start + d] += result.gradient[d];
                gradient[p1_dof_start + d] += result.gradient[d + 3];
            }

            // Assemble Hessian from blocks
            this.bsm.addBlockAt(p0_block_idx, p0_block_idx, result.hessian.H_p0p0);
            this.bsm.addBlockAt(p0_block_idx, p1_block_idx, result.hessian.H_p0p1);
            this.bsm.addBlockAt(p1_block_idx, p1_block_idx, result.hessian.H_p1p1);
        }

        // Step 3: Process bending forces
        for (let i = 1; i < this.getNumVertices() - 1; i++) {
            const result = this.computeBendingForces(i, this.bendStiffness);

            const p_prev_block_idx = 2 * (i - 1);
            const p_curr_block_idx = 2 * i;
            const p_next_block_idx = 2 * (i + 1);

            const p_prev_dof_start = this.boundaries[p_prev_block_idx];
            const p_curr_dof_start = this.boundaries[p_curr_block_idx];
            const p_next_dof_start = this.boundaries[p_next_block_idx];

            // Assemble gradient
            const dof_indices = [p_prev_dof_start, p_curr_dof_start, p_next_dof_start];
            for (let j = 0; j < 3; j++) {
                for (let d = 0; d < 3; d++) {
                    gradient[dof_indices[j] + d] += result.gradient[j * 3 + d];
                }
            }

            // Assemble Hessian
            const block_indices = [p_prev_block_idx, p_curr_block_idx, p_next_block_idx];
            for (let j = 0; j < 3; j++) {
                for (let k = j; k < 3; k++) { // SymmetricBlockSparseMatrix: only add for k >= j
                    this.bsm.addBlockAt(block_indices[j], block_indices[k], result.hessian[j][k]);
                }
            }
        }

        // Step 4: Add mass matrix and gravity
        const inv_dt2 = 1.0 / (deltaTime * deltaTime);
        const mass_term = this.vertexMass * inv_dt2;
        const inertia_term = this.vertexMass * this.radius * this.radius * inv_dt2;

        for (let i = 0; i < this.getNumVertices(); i++) {
            const posBlockIndex = 2 * i;
            const thetaBlockIndex = 2 * i + 1;

            // Add mass to position block diagonal
            const massMatrix = Matrix.identity(3).scale(mass_term);
            this.bsm.addBlockAt(posBlockIndex, posBlockIndex, massMatrix);

            // Add inertia to theta block diagonal
            const inertiaMatrix = Matrix.identity(1).scale(inertia_term);
            this.bsm.addBlockAt(thetaBlockIndex, thetaBlockIndex, inertiaMatrix);

            // Add gravity force to gradient (dE/dx = -f_gravity)
            const posDofStart = this.boundaries[posBlockIndex];
            gradient[posDofStart + 1] += this.vertexMass * -this.gravity.y;
        }

        // Step 5: Handle fixed blocks
        for (const blockIndex of this.fixedBlockIndices) {
            this.bsm.setFixedBlock(blockIndex);
            
            // Zero out corresponding gradient entries
            const dofStart = this.boundaries[blockIndex];
            const dofEnd = this.boundaries[blockIndex + 1];
            for (let i = dofStart; i < dofEnd; i++) {
                gradient[i] = 0;
            }
        }

        // Step 6: Solve linear system: (H + M/dt^2) * delta_q = gradient
        const delta_q = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);

        // Step 7: Update velocities and positions
        for (let i = 0; i < numDofs; i++) {
            if (!this.fixedDofs.has(i)) {
                // Update position: q_new = q_pred - delta_q
                this.q[i] -= delta_q[i];
                // Update velocity: v_new = (q_new - q_old) / dt = v_old - delta_q / dt
                this.velocities[i] -= delta_q[i] / deltaTime;
            }
        }
    }

    // Public method to get current positions for rendering
    public getPositions(): Vector3[] {
        const positions: Vector3[] = [];
        const numVertices = (this.boundaries.length - 1) / 2;
        for (let i = 0; i < numVertices; i++) {
            positions.push(this.getPosition(i));
        }
        return positions;
    }

    public getEdges(): number[][] {
        const edges: number[][] = [];
        const numVertices = this.getNumVertices();
        for (let i = 0; i < numVertices - 1; i++) {
            edges.push([i, i + 1]);
        }
        return edges;
    }
}

/**
 * Utility function to create a simple rod geometry.
 */
export function createRod(
    numVertices: number, 
    length: number = 5.0,
    stretchStiffness: number = 1000,
    bendStiffness: number = 100,
    twistStiffness: number = 100,
    vertexMass: number = 1.0,
    radius: number = 0.1
): DERGeometry {
    const positions: Vector3[] = [];
    
    // Create straight rod along y-axis, starting from (0,0,0)
    for (let i = 0; i < numVertices; i++) {
        const x = (i / (numVertices - 1)) * length;
        positions.push(new Vector3(x, 5, 0));
    }
    
    const fixedVertices = new Set<number>();
    fixedVertices.add(0); // Fix one end of the rod
    
    const geometry: DERGeometry = {
        positions,
        fixedVertices,
        stretchStiffness,
        bendStiffness,
        twistStiffness,
        vertexMass,
        radius
    };
    
    return geometry;
}

/**
 * Utility function to create an L-shaped rod geometry parallel to the ground.
 */
export function createLShapedRod(
    numVertices: number = 21, 
    length: number = 5.0,
    stretchStiffness: number = 1000,
    bendStiffness: number = 100,
    twistStiffness: number = 100,
    vertexMass: number = 1.0,
    radius: number = 0.1
): DERGeometry {
    const positions: Vector3[] = [];
    const bendIndex = Math.floor(numVertices / 2);
    const segmentLength = length / 2;
    const y_level = 5.0; // Height above the ground

    // First segment (along X-axis)
    for (let i = 0; i <= bendIndex; i++) {
        const x = (i / bendIndex) * segmentLength;
        positions.push(new Vector3(x, y_level, 0));
    }

    // Second segment (along Z-axis)
    for (let i = 1; i < numVertices - bendIndex; i++) {
        const z = (i / (numVertices - bendIndex - 1)) * segmentLength;
        positions.push(new Vector3(segmentLength, y_level, z));
    }
    
    const fixedVertices = new Set<number>();
    fixedVertices.add(0); // Fix one end of the rod
    
    const geometry: DERGeometry = {
        positions,
        fixedVertices,
        stretchStiffness,
        bendStiffness,
        twistStiffness,
        vertexMass,
        radius
    };
    
    return geometry;
}