import { Vector3 } from "@babylonjs/core";

// 3x3 matrix for block operations
class Matrix3x3 {
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
}

// Block sparse matrix using CSR format
class BlockSparseMatrix {
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

// Spring energy, gradient, and hessian computation
function springEnergyGradientHessian(
    pos0: Vector3, 
    pos1: Vector3, 
    restLength: number, 
    stiffness: number
): { energy: number, gradients: Vector3[], hessian: Matrix3x3[][] } {
    const diff = pos1.subtract(pos0);
    const currentLength = diff.length();
    
    if (currentLength < 1e-8) {
        // Degenerate case
        return {
            energy: 0,
            gradients: [new Vector3(0, 0, 0), new Vector3(0, 0, 0)],
            hessian: [[Matrix3x3.identity().scale(stiffness), Matrix3x3.identity().scale(-stiffness)], 
                     [Matrix3x3.identity().scale(-stiffness), Matrix3x3.identity().scale(stiffness)]]
        };
    }
    
    const strain = currentLength - restLength; 
    const energy = 0.5 * stiffness * strain * strain;
    
    const u = diff.normalize();  
    const gradients = [u.scale(-stiffness * strain), u.scale(stiffness * strain)];
    
    // Hessian computation
    const uu = Matrix3x3.outerProduct(u, u);
    const H = uu.scale(stiffness); 
    
    const hessian = [
        [H, H.scale(-1)],
        [H.scale(-1), H]
    ];
    
    return { energy, gradients, hessian };
}

export interface Geometry {
    positions: Vector3[];
    edges: number[]; // [v0, v1, v2, v3, ...] pairs
    fixedVertices: Set<number>;
    stiffness: number; // Material property
    vertexMass: number; // Mass per vertex
}

export class ImplicitSolver {
    private positions: Vector3[];
    private edges: number[];
    private fixedVertices: Set<number>;
    private restLengths: number[];
    private stiffness: number;
    private vertexMass: number;
    
    private bsm: BlockSparseMatrix;
    private velocities: Vector3[];
    private gravity: Vector3 = new Vector3(0, -10.0, 0);

    constructor(geometry: Geometry) {
        this.positions = geometry.positions.map(p => p.clone());
        this.edges = [...geometry.edges];
        this.fixedVertices = new Set(geometry.fixedVertices);
        this.stiffness = geometry.stiffness;
        this.vertexMass = geometry.vertexMass;
        this.restLengths = new Array(this.getNumEdges());
        for (let i = 0; i < this.getNumEdges(); i++) {
            const v0 = this.edges[i * 2];
            const v1 = this.edges[i * 2 + 1];
            this.restLengths[i] = Vector3.Distance(geometry.positions[v0], geometry.positions[v1]);
        }
        
        this.bsm = new BlockSparseMatrix();
        
        // Create sparse matrix structure
        const structure = this.createSparseMatrixStructure();
        this.bsm.initialize(structure.row2idx, structure.idx2col);
        
        // Initialize velocities to zero
        this.velocities = new Array(this.getNumPositions())
            .fill(null).map(() => new Vector3(0, 0, 0));
    }

    public getNumPositions(): number {
        return this.positions.length;
    }

    public getNumEdges(): number {
        return this.edges.length / 2;
    }

    public getPositions(): Vector3[] {
        return this.positions.map(p => p.clone());
    }

   public getEdges(): [number, number][] {
        const edgePairs: [number, number][] = [];
        for (let i = 0; i < this.edges.length; i += 2) {
            edgePairs.push([this.edges[i], this.edges[i + 1]]);
        }
        return edgePairs;
    }

    private isFixed(vertexIndex: number): boolean {
        return this.fixedVertices.has(vertexIndex);
    }

    private createSparseMatrixStructure(): { row2idx: number[], idx2col: number[] } {
        const numVertices = this.getNumPositions();
        const adjacency = new Map<number, Set<number>>();
        
        // Initialize adjacency lists
        for (let i = 0; i < numVertices; i++) {
            adjacency.set(i, new Set([i])); // Self-connection
        }
        
        // Add edge connections
        for (let i = 0; i < this.getNumEdges(); i++) {
            const v0 = this.edges[i * 2];
            const v1 = this.edges[i * 2 + 1];
            adjacency.get(v0)!.add(v1);
            adjacency.get(v1)!.add(v0);
        }
        
        // Build CSR structure
        const row2idx: number[] = [0];
        const idx2col: number[] = [];
        
        for (let i = 0; i < numVertices; i++) {
            const neighbors = Array.from(adjacency.get(i)!).sort((a, b) => a - b);
            for (const neighbor of neighbors) {
                idx2col.push(neighbor);
            }
            row2idx.push(idx2col.length);
        }
        
        return { row2idx, idx2col };
    }

    step(deltaTime: number): void {
        const numVertices = this.getNumPositions();
        const gradient = new Array(numVertices).fill(null).map(() => new Vector3(0, 0, 0));
        
        this.bsm.setZero();
        
        // Step 1: Update positions using current velocities
        for (let i = 0; i < numVertices; i++) {
            if (!this.isFixed(i)) {
                this.positions[i].addInPlace(this.velocities[i].scale(deltaTime));
            }
        }
        
        let totalEnergy = 0;
        
        // Step 2: Process each edge (spring) to compute forces and hessian
        for (let i = 0; i < this.getNumEdges(); i++) {
            const v0 = this.edges[i * 2];
            const v1 = this.edges[i * 2 + 1];
            const restLength = this.restLengths[i];
            
            const result = springEnergyGradientHessian(
                this.positions[v0],
                this.positions[v1],
                restLength,
                this.stiffness
            );
            
            totalEnergy += result.energy;
            
            // Add gradients (forces)
            gradient[v0].addInPlace(result.gradients[0]);
            gradient[v1].addInPlace(result.gradients[1]);
            
            // Add hessian blocks to matrix
            this.bsm.addBlockAt(v0, v0, result.hessian[0][0]);
            this.bsm.addBlockAt(v0, v1, result.hessian[0][1]);
            this.bsm.addBlockAt(v1, v0, result.hessian[1][0]);
            this.bsm.addBlockAt(v1, v1, result.hessian[1][1]);
        }
         // Step 3: Add mass matrix (mass_point / (timeStep * timeStep))
        const massMatrix = Matrix3x3.identity().scale(this.vertexMass / (deltaTime * deltaTime));
        for (let i = 0; i < numVertices; i++) {
            this.bsm.addBlockAt(i, i, massMatrix);
        }

        // Step 4: Add gravity forces
        for (let i = 0; i < numVertices; i++) {
            gradient[i].subtractInPlace(this.gravity.scale(this.vertexMass));
            totalEnergy -= this.vertexMass * Vector3.Dot(this.gravity, this.positions[i]);
        }
        
        // Step 5: Handle fixed vertices
        for (const fixedIndex of this.fixedVertices) {
            gradient[fixedIndex] = new Vector3(0, 0, 0);
            this.velocities[fixedIndex] = new Vector3(0, 0, 0);
            this.bsm.setFixed(fixedIndex);
        }
        
        // Step 6: Solve linear system
        const delta = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);
        
        // Step 7: Update velocities and positions
        for (let i = 0; i < numVertices; i++) {
            if (!this.isFixed(i)) {
                this.velocities[i].subtractInPlace(delta[i].scale(1 / deltaTime));
                this.positions[i].subtractInPlace(delta[i]);
            }
        }
    }
}

// Utility function to create a chain of springs
export function createChain(length: number, resolution: number, stiffness: number = 100, vertexMass: number = 1.0): Geometry {
    const positions = [];
    const edges = [];
    
    // Create horizontal chain along x-axis
    for (let i = 0; i < resolution; i++) {
        positions.push(new Vector3((i / resolution) * length, 5, 0));  // Horizontal chain at y=5
    }
    
    for (let i = 0; i < resolution - 1; i++) {
        edges.push(i, i + 1);
    }
    
    const geometry = {
        positions: positions,
        edges: edges,
        fixedVertices: new Set([0]),
        stiffness: stiffness,
        vertexMass: vertexMass
    };
    
    return geometry;
}

export function createChainwithBend(length: number, resolution: number, stiffness: number = 100, vertexMass: number = 1.0): Geometry {
    const positions = [];
    const edges = [];

    // Create horizontal chain along x-axis
    for (let i = 0; i < resolution; i++) {
        positions.push(new Vector3((i / resolution) * length, 5, 0));  // Horizontal chain at y=5
    }

    for (let i = 0; i < resolution - 1; i++) {
        edges.push(i, i + 1);
    }

    // Add bending springs (every second edge)
    for (let i = 0; i < resolution - 2; i += 2) {
        edges.push(i, i + 2); // Bend connection
    }

    const geometry = {
        positions: positions,
        edges: edges,
        fixedVertices: new Set([0]),
        stiffness: stiffness,
        vertexMass: vertexMass
    };

    return geometry;
}

// Utility function to create a cloth (2D grid of springs)
export function createCloth(width: number, height: number, resolutionX: number, resolutionY: number, stiffness: number = 100, vertexMass: number = 1.0): Geometry {
    const positions = [];
    const edges = [];
    
    // Create grid of vertices
    for (let j = 0; j <= resolutionY; j++) {
        for (let i = 0; i <= resolutionX; i++) {
            const x = (i / resolutionX) * width - width / 2;  // Center the cloth
            const y = 5;  // Height of the cloth
            const z = (j / resolutionY) * height - height / 2;  // Center the cloth
            positions.push(new Vector3(x, y, z));
        }
    }
    
    // Create horizontal edges (structural springs)
    for (let j = 0; j <= resolutionY; j++) {
        for (let i = 0; i < resolutionX; i++) {
            const v0 = j * (resolutionX + 1) + i;
            const v1 = j * (resolutionX + 1) + i + 1;
            edges.push(v0, v1);
        }
    }
    
    // Create vertical edges (structural springs)
    for (let j = 0; j < resolutionY; j++) {
        for (let i = 0; i <= resolutionX; i++) {
            const v0 = j * (resolutionX + 1) + i;
            const v1 = (j + 1) * (resolutionX + 1) + i;
            edges.push(v0, v1);
        }
    }
    
    // Create diagonal edges (shear springs) - optional for better cloth behavior
    for (let j = 0; j < resolutionY; j++) {
        for (let i = 0; i < resolutionX; i++) {
            const topLeft = j * (resolutionX + 1) + i;
            const topRight = j * (resolutionX + 1) + i + 1;
            const bottomLeft = (j + 1) * (resolutionX + 1) + i;
            const bottomRight = (j + 1) * (resolutionX + 1) + i + 1;
            
            // Diagonal from top-left to bottom-right
            edges.push(topLeft, bottomRight);
            // Diagonal from top-right to bottom-left
            edges.push(topRight, bottomLeft);
        }
    }
    
    const geometry = {
        positions: positions,
        edges: edges,
        fixedVertices: new Set([0, resolutionX]),
        stiffness: stiffness,
        vertexMass: vertexMass
    };
    
    return geometry;
}
