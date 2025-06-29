import { Vector3 } from "@babylonjs/core";
import { SymmetricBlockSparseMatrix, Matrix } from "./bsm";

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

        // 3. Initialize the Symmetric Block Sparse Matrix
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

    // Helper to get a position vector from the flat q array
    private getPosition(vertexIndex: number): Vector3 {
        const blockIndex = 2 * vertexIndex;
        const dofIndex = this.boundaries[blockIndex];
        return new Vector3(this.q[dofIndex], this.q[dofIndex + 1], this.q[dofIndex + 2]);
    }

    // Helper to get a theta value from the flat q array
    private getTheta(vertexIndex: number): number {
        const blockIndex = 2 * vertexIndex + 1;
        const dofIndex = this.boundaries[blockIndex];
        return this.q[dofIndex];
    }

    private getNumVertices(): number {
        return (this.boundaries.length - 1) / 2;
    }

    private getNumElements(): number {
        return this.getNumVertices() - 1;
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

        // Step 2: Process stretching forces
        for (let i = 0; i < this.getNumElements(); i++) {
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

        // Step 3: Add mass matrix and gravity
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

        // Step 4: Handle fixed blocks
        for (const blockIndex of this.fixedBlockIndices) {
            this.bsm.setFixedBlock(blockIndex);
            
            // Zero out corresponding gradient entries
            const dofStart = this.boundaries[blockIndex];
            const dofEnd = this.boundaries[blockIndex + 1];
            for (let i = dofStart; i < dofEnd; i++) {
                gradient[i] = 0;
            }
        }

        // Step 5: Solve linear system: (H + M/dt^2) * delta_q = gradient
        const delta_q = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);

        // Step 6: Update velocities and positions
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
