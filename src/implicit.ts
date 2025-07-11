import { Vector3 } from "@babylonjs/core";
import { Matrix3x3 } from "./util";
import { Geometry } from "./geometry";
import { Params } from "./params";

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

export class ImplicitSolver {
    numVertices: number;
    pos: Float32Array;
    prevPos: Float32Array;
    vel: Float32Array;
    masses: Float32Array;
    fixedVertices: Uint8Array;

    numEdges: number;
    edges: Uint16Array;
    stiffnesses: Float32Array;
    restLengths: Float32Array;

    bsm: BlockSparseMatrix;

    params: Params;

    constructor(geometry: Geometry, params: Params) {
        this.numVertices = geometry.pos.length / 3;
        this.pos = new Float32Array(geometry.pos);
        this.prevPos = new Float32Array(this.numVertices * 3);
        this.vel = new Float32Array(this.numVertices * 3);
        this.masses = new Float32Array(geometry.masses);
        this.fixedVertices = new Uint8Array(geometry.fixedVertices);
        this.numEdges = geometry.edges.length / 2;
        this.edges = new Uint16Array(geometry.edges);
        this.stiffnesses = new Float32Array(geometry.stiffnesses);
        this.restLengths = new Float32Array(this.numEdges);
        this.params = params;

        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            
            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);

            this.restLengths[e] = Vector3.Distance(p0, p1);
        }
        
        this.bsm = new BlockSparseMatrix();
        
        // Create sparse matrix structure
        const structure = this.createSparseMatrixStructure();
        this.bsm.initialize(structure.row2idx, structure.idx2col);
    }

    private createSparseMatrixStructure(): { row2idx: number[], idx2col: number[] } {
        const adjacency = new Map<number, Set<number>>();
        
        // Initialize adjacency lists
        for (let i = 0; i < this.numVertices; i++) {
            adjacency.set(i, new Set([i])); // Self-connection
        }
        
        // Add edge connections
        for (let i = 0; i < this.numEdges; i++) {
            const v0 = this.edges[i * 2];
            const v1 = this.edges[i * 2 + 1];
            adjacency.get(v0)!.add(v1);
            adjacency.get(v1)!.add(v0);
        }
        
        // Build CSR structure
        const row2idx: number[] = [0];
        const idx2col: number[] = [];
        
        for (let i = 0; i < this.numVertices; i++) {
            const neighbors = Array.from(adjacency.get(i)!).sort((a, b) => a - b);
            for (const neighbor of neighbors) {
                idx2col.push(neighbor);
            }
            row2idx.push(idx2col.length);
        }
        
        return { row2idx, idx2col };
    }

    step(): void {
        const g = this.params.gravity;
        const dt = this.params.dt;
        const invDt2 = 1 / (dt * dt);

        const gradient = new Array(this.numVertices).fill(null).map(() => new Vector3(0, 0, 0));

        this.bsm.setZero();

        this.prevPos.set(this.pos);
        
        // Step 1: Update positions using current velocities
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;
            const p = this.getVector3(this.pos, i);
            const v = this.getVector3(this.vel, i);
            this.setVector3(this.pos, i, p.add(v.scale(dt)));
        }
        
        let totalEnergy = 0;
        
        // Step 2: Process each edge (spring) to compute forces and hessian
        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);
            const restLength = this.restLengths[e];
            const stiffness = this.stiffnesses[e];

            const result = springEnergyGradientHessian(
                p0,
                p1,
                restLength,
                stiffness
            );
            
            totalEnergy += result.energy;
            
            // Add gradients (forces)
            gradient[id0].addInPlace(result.gradients[0]);
            gradient[id1].addInPlace(result.gradients[1]);

            // Add hessian blocks to matrix
            this.bsm.addBlockAt(id0, id0, result.hessian[0][0]);
            this.bsm.addBlockAt(id0, id1, result.hessian[0][1]);
            this.bsm.addBlockAt(id1, id0, result.hessian[1][0]);
            this.bsm.addBlockAt(id1, id1, result.hessian[1][1]);
        }

         // Step 3: Add mass matrix (mass_point / (timeStep * timeStep))
        for (let i = 0; i < this.numVertices; i++) {
            const mass = this.masses[i];
            const massMatrix = Matrix3x3.identity().scale(mass / (dt * dt));
            this.bsm.addBlockAt(i, i, massMatrix);
        }

        // Step 4: Add gravity forces
        for (let i = 0; i < this.numVertices; i++) {
            const mass = this.masses[i];
            const p = this.getVector3(this.pos, i);
            gradient[i].subtractInPlace(g.scale(mass));
            totalEnergy -= mass * Vector3.Dot(g, p);
        }
        
        // Step 5: Handle fixed vertices
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) {
                gradient[i] = new Vector3(0, 0, 0);
                this.setVector3(this.vel, i, new Vector3(0, 0, 0));
                this.bsm.setFixed(i);
            }
        }
        
        // Step 6: Solve linear system
        const delta = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);
        
        // Step 7: Update velocities and positions
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;
            let p = this.getVector3(this.pos, i);
            p = p.subtract(delta[i]);
            this.setVector3(this.pos, i, p);

            const prevP = this.getVector3(this.prevPos, i);
            const v = p.subtract(prevP).scale(1 / dt);
            this.setVector3(this.vel, i, v);
        }
    }

    getVector3(array: Float32Array, i: number): Vector3 {
        return new Vector3(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
    }

    setVector3(array: Float32Array, i: number, v: Vector3): void {
        array[i * 3] = v.x; array[i * 3 + 1] = v.y; array[i * 3 + 2] = v.z;
    }
}
