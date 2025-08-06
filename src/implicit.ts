import { Vector3 } from "@babylonjs/core";
import { Matrix3x3, BlockSparseMatrix } from "./util";
import { Geometry } from "./geometry";
import { Params } from "./params";

function springEnergy(pos0: Vector3, pos1: Vector3, restLength: number, stiffness: number): number {
    const diff = pos1.subtract(pos0);
    const length = diff.length();
    const C = length - restLength;
    return 0.5 * stiffness * C * C;
}

function springGradient(pos0: Vector3, pos1: Vector3, restLength: number, stiffness: number): Vector3[] {
    const diff = pos1.subtract(pos0);
    const length = diff.length();
    if (length < 1e-8) return [new Vector3(0, 0, 0), new Vector3(0, 0, 0)]; // Avoid division by zero
    const C = length - restLength;
    const u01 = diff.normalize();
    const dC: Vector3[] = [u01.scale(-1), u01];
    return [dC[0].scale(stiffness * C), dC[1].scale(stiffness * C)];
}

function springHessian(pos0: Vector3, pos1: Vector3, restLength: number, stiffness: number): Matrix3x3[][] {
    const diff = pos1.subtract(pos0);
    const length = diff.length();
    if (length < 1e-8) {
        const I = Matrix3x3.identity();
        return [[I.scale(stiffness), I.scale(-stiffness)], [I.scale(-stiffness), I.scale(stiffness)]];
    }
    const C = length - restLength;
    const u01 = diff.normalize();
    const uu = Matrix3x3.outerProduct(u01, u01);
    const o = uu.scale(stiffness).add((Matrix3x3.identity().subtract(uu)).scale(stiffness * C / length));
    return [[o, o.scale(-1)], [o.scale(-1), o]];
}

export class ImplicitSolver {
    numVertices: number;
    pos: Float32Array;
    prevPos: Float32Array;
    inertiaPos: Float32Array;
    vel: Float32Array;
    masses: Float32Array;
    fixedVertices: Uint8Array;

    numEdges: number;
    edges: Uint32Array;
    stiffnesses: Float32Array;
    restLengths: Float32Array;

    bsm: BlockSparseMatrix;

    params: Params;

    constructor(geometry: Geometry, params: Params) {
        this.numVertices = geometry.pos.length / 3;
        this.pos = new Float32Array(geometry.pos);
        this.prevPos = new Float32Array(this.numVertices * 3);
        this.inertiaPos = new Float32Array(this.numVertices * 3);
        this.vel = new Float32Array(this.numVertices * 3);
        this.masses = new Float32Array(geometry.masses);
        this.fixedVertices = new Uint8Array(geometry.fixedVertices);
        this.numEdges = geometry.edges.length / 2;
        this.edges = new Uint32Array(geometry.edges);
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
        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            adjacency.get(id0)!.add(id1);
            adjacency.get(id1)!.add(id0);
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
        for (let step = 0; step < this.params.numSubsteps; step++) {
            this.solve();
        }
    }

    solve(): void {
        this.prevPos.set(this.pos);

        const g = this.params.g;
        const dt = this.params.dt / this.params.numSubsteps;
        const invDt = 1 / dt;
        const invDt2 = 1 / (dt * dt);

        // Compute inertia positions and warm start
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) this.setVector3(this.inertiaPos, i, this.getVector3(this.pos, i));
            else {
                let p = this.getVector3(this.pos, i);
                let v = this.getVector3(this.vel, i);
                // v.addInPlace(g.scale(dt)); // Apply gravity
                p.addInPlace(v.scale(dt));
                this.setVector3(this.inertiaPos, i, p);
                this.setVector3(this.pos, i, p);
            }
        }

        for (let itr = 0; itr < this.params.numIterations; itr++) {
            const gradient = new Array(this.numVertices).fill(null).map(() => new Vector3(0, 0, 0));
            this.bsm.setZero();
            
            let totalEnergy = 0;
            
            // Process each edge (spring) to compute forces and hessian
            for (let e = 0; e < this.numEdges; e++) {
                const id0 = this.edges[e * 2];
                const id1 = this.edges[e * 2 + 1];
                const p0 = this.getVector3(this.pos, id0);
                const p1 = this.getVector3(this.pos, id1);
                const restLength = this.restLengths[e];
                const stiffness = this.stiffnesses[e];

                // Compute spring energy, gradient and hessian
                const energy = springEnergy(p0, p1, restLength, stiffness);
                const gradients = springGradient(p0, p1, restLength, stiffness);
                const hessian = springHessian(p0, p1, restLength, stiffness);

                totalEnergy += energy;

                // Add gradients (forces)
                gradient[id0].addInPlace(gradients[0]);
                gradient[id1].addInPlace(gradients[1]);

                // Add hessian blocks to matrix
                this.bsm.addBlockAt(id0, id0, hessian[0][0]);
                this.bsm.addBlockAt(id0, id1, hessian[0][1]);
                this.bsm.addBlockAt(id1, id0, hessian[1][0]);
                this.bsm.addBlockAt(id1, id1, hessian[1][1]);
            }

            // Add inertia term
            for (let i = 0; i < this.numVertices; i++) {
                const mass = this.masses[i];
                const p = this.getVector3(this.pos, i);
                const inertiaP = this.getVector3(this.inertiaPos, i);
                gradient[i].addInPlace(p.subtract(inertiaP).scale(mass * invDt2));
                totalEnergy += 0.5 * mass * invDt2 * Vector3.DistanceSquared(inertiaP, p);
            }

            // Add gravity forces
            for (let i = 0; i < this.numVertices; i++) {
                const mass = this.masses[i];
                const p = this.getVector3(this.pos, i);
                gradient[i].subtractInPlace(g.scale(mass));
                totalEnergy -= mass * Vector3.Dot(g, p);
            }

            // Add mass matrix (mass_point / (timeStep * timeStep))
            for (let i = 0; i < this.numVertices; i++) {
                const mass = this.masses[i];
                const massMatrix = Matrix3x3.identity().scale(mass * invDt2);
                this.bsm.addBlockAt(i, i, massMatrix);
            }
            
            // Handle fixed vertices
            for (let i = 0; i < this.numVertices; i++) {
                if (this.fixedVertices[i]) {
                    gradient[i] = new Vector3(0, 0, 0);
                    this.setVector3(this.vel, i, new Vector3(0, 0, 0));
                    this.bsm.setFixed(i);
                }
            }
            
            // Solve linear system
            const delta = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);

            // Update positions
            for (let i = 0; i < this.numVertices; i++) {
                // if (this.fixedVertices[i]) continue;
                const p = this.getVector3(this.pos, i);
                const d = delta[i]; // use alpha if needed
                this.setVector3(this.pos, i, p.subtract(d));
            }

            if (itr === this.params.numIterations - 1) console.log("Total energy:", totalEnergy);
        }
            
        // Update velocities
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;
            const p = this.getVector3(this.pos, i);
            const prevP = this.getVector3(this.prevPos, i);
            const v = p.subtract(prevP).scale(invDt);
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
