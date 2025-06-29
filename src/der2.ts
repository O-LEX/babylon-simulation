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
    private gravity: Vector3 = new Vector3(0, -9.81, 0);

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
        vertexIndex: number, // This is the index `i` of the central vertex x_i
        stiffness: number
    ): {
        energy: number;
        gradient: number[]; // A 9-element array for forces on x_{i-1}, x_i, x_{i+1}
        hessianBlocks: Matrix[][]; // 3x3 grid of 3x3 matrices
    } {
        // Get positions of the three vertices forming the bend
        const x_prev = this.getPosition(vertexIndex - 1);
        const x_curr = this.getPosition(vertexIndex);
        const x_next = this.getPosition(vertexIndex + 1);

        // Edge vectors
        const e_prev = x_curr.subtract(x_prev); // e_{i-1}
        const e_curr = x_next.subtract(x_curr); // e_i

        const l_prev = e_prev.length();
        const l_curr = e_curr.length();

        const zeros = {
            energy: 0,
            gradient: [0, 0, 0, 0, 0, 0, 0, 0, 0],
            hessianBlocks: Array(3).fill(null).map(() => Array(3).fill(new Matrix(3, 3)))
        };

        if (l_prev < 1e-8 || l_curr < 1e-8) {
            return zeros;
        }

        // Tangents (already computed)
        const t_prev = this.tangents[vertexIndex - 1];
        const t_curr = this.tangents[vertexIndex];

        // Curvature binormal
        const chi = 1.0 + Vector3.Dot(t_prev, t_curr);
        if (Math.abs(chi) < 1e-8) {
            return zeros;
        }
        const kappa_b = Vector3.Cross(t_prev, t_curr).scale(2.0 / chi);

        // Material frames (already computed)
        const d1_prev = this.mat_d1[vertexIndex - 1];
        const d2_prev = this.mat_d2[vertexIndex - 1];
        const d1_curr = this.mat_d1[vertexIndex];
        const d2_curr = this.mat_d2[vertexIndex];

        // Material curvature components
        const kappa1 = 0.5 * Vector3.Dot(d2_prev.add(d2_curr), kappa_b);
        const kappa2 = -0.5 * Vector3.Dot(d1_prev.add(d1_curr), kappa_b);

        // Rest curvature (assumed zero for now)
        const kappa1_rest = 0.0;
        const kappa2_rest = 0.0;

        const kappa1_diff = kappa1 - kappa1_rest;
        const kappa2_diff = kappa2 - kappa2_rest;

        // Voronoi length
        const voronoi_length = (l_prev + l_curr) * 0.5;
        if (voronoi_length < 1e-8) {
            return zeros;
        }

        // Bending energy
        const energy_factor = stiffness / voronoi_length;
        const energy = 0.5 * energy_factor * (kappa1_diff * kappa1_diff + kappa2_diff * kappa2_diff);

        // --- Gradients and Hessians (complex part, adapted from der.ts) ---

        const dE_dKappa1 = energy_factor * kappa1_diff;
        const dE_dKappa2 = energy_factor * kappa2_diff;

        const t_tilde = t_prev.add(t_curr).scale(1.0 / chi);
        const d1_tilde = d1_prev.add(d1_curr).scale(0.5);
        const d2_tilde = d2_prev.add(d2_curr).scale(0.5);

        const dKappa1_de_prev = t_tilde.scale(-kappa1).add(Vector3.Cross(t_curr, d2_tilde)).scale(1.0 / l_prev);
        const dKappa1_de_curr = t_tilde.scale(-kappa1).subtract(Vector3.Cross(t_prev, d2_tilde)).scale(1.0 / l_curr);

        const dKappa2_de_prev = t_tilde.scale(-kappa2).subtract(Vector3.Cross(t_curr, d1_tilde)).scale(1.0 / l_prev);
        const dKappa2_de_curr = t_tilde.scale(-kappa2).add(Vector3.Cross(t_prev, d1_tilde)).scale(1.0 / l_curr);

        const dE_de_prev = dKappa1_de_prev.scale(dE_dKappa1).add(dKappa2_de_prev.scale(dE_dKappa2));
        const dE_de_curr = dKappa1_de_curr.scale(dE_dKappa1).add(dKappa2_de_curr.scale(dE_dKappa2));

        const grad_x_prev = dE_de_prev.scale(-1.0);
        const grad_x_curr = dE_de_prev.subtract(dE_de_curr);
        const grad_x_next = dE_de_curr;

        const gradient = [
            grad_x_prev.x, grad_x_prev.y, grad_x_prev.z,
            grad_x_curr.x, grad_x_curr.y, grad_x_curr.z,
            grad_x_next.x, grad_x_next.y, grad_x_next.z
        ];

        // Simplified Hessian (first-order approximation)
        const H00 = createOuterProductMatrix(grad_x_prev, grad_x_prev).scale(1.0 / stiffness); // Approximation
        const H11 = createOuterProductMatrix(grad_x_curr, grad_x_curr).scale(1.0 / stiffness);
        const H22 = createOuterProductMatrix(grad_x_next, grad_x_next).scale(1.0 / stiffness);
        const H01 = createOuterProductMatrix(grad_x_prev, grad_x_curr).scale(1.0 / stiffness);
        const H02 = createOuterProductMatrix(grad_x_prev, grad_x_next).scale(1.0 / stiffness);
        const H12 = createOuterProductMatrix(grad_x_curr, grad_x_next).scale(1.0 / stiffness);

        const hessianBlocks = [
            [H00, H01, H02],
            [H01, H11, H12], // Symmetric
            [H02, H12, H22]  // Symmetric
        ];

        return { energy, gradient, hessianBlocks };
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
        hessianBlocks: { H_p0p0: Matrix, H_p0p1: Matrix, H_p1p1: Matrix } 
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
                hessianBlocks: {
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

        // Hessian blocks are 3x3 matrices
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
            hessianBlocks: {
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
            this.bsm.addBlockAt(p0_block_idx, p0_block_idx, result.hessianBlocks.H_p0p0);
            this.bsm.addBlockAt(p0_block_idx, p1_block_idx, result.hessianBlocks.H_p0p1);
            this.bsm.addBlockAt(p1_block_idx, p1_block_idx, result.hessianBlocks.H_p1p1);
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
                    this.bsm.addBlockAt(block_indices[j], block_indices[k], result.hessianBlocks[j][k]);
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
