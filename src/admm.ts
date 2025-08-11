import { Vector3 } from "@babylonjs/core";
import { Triplet, SparseMatrix } from "./util";
import { Geometry } from "./geometry";
import { Params } from "./params";

interface EnergyTerm {
    offset: number;
    getId(): number[];
    getD(): Triplet[];
    getW(): number[];
    update(pos: Float32Array, z: Float32Array, u: Float32Array): void;
}

class SpringEnergyTerm implements EnergyTerm {
    offset: number;
    private id0: number;
    private id1: number;
    private stiffness: number;
    private restLength: number;

    constructor(id0: number, id1: number, stiffness: number, restLength: number) {
        this.offset = 0;
        this.id0 = id0;
        this.id1 = id1;
        this.stiffness = stiffness;
        this.restLength = restLength;
    }

    update(pos: Float32Array, z: Float32Array, u: Float32Array): void {
        let z_i = new Vector3(z[this.offset], z[this.offset + 1], z[this.offset + 2]);
        let u_i = new Vector3(u[this.offset], u[this.offset + 1], u[this.offset + 2]);
        const p0 = new Vector3(pos[this.id0 * 3], pos[this.id0 * 3 + 1], pos[this.id0 * 3 + 2]);
        const p1 = new Vector3(pos[this.id1 * 3], pos[this.id1 * 3 + 1], pos[this.id1 * 3 + 2]);
        let q = p1.subtract(p0); // Dix
        q.addInPlace(u_i); // Dix + ui
        const q_length = q.length();
        const p_i = q.scale(this.restLength / q_length);
        z_i = p_i.add(q).scale(0.5);
        u_i = q.subtract(z_i);
        z[this.offset] = z_i.x;
        z[this.offset + 1] = z_i.y;
        z[this.offset + 2] = z_i.z;
        u[this.offset] = u_i.x;
        u[this.offset + 1] = u_i.y;
        u[this.offset + 2] = u_i.z;
    }

    getId(): number[] {
        return [this.id0, this.id1];
    }

    getD(): Triplet[] {
        const D = [
            [-1, 0, 0, 1, 0, 0],
            [0, -1, 0, 0, 1, 0],
            [0, 0, -1, 0, 0, 1],
        ];
        const triplets: Triplet[] = [];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 6; j++) {
                triplets.push({ row: i, col: j, val: D[i][j] });
            }
        }
        return triplets;
    }

    getW(): number[] {
        const weight = Math.sqrt(this.stiffness);
        return [weight, weight, weight];
    }
}

class FixedEnergyTerm implements EnergyTerm {
    offset: number;
    private id: number;
    private pos: Vector3;

    constructor(id: number, pos: Vector3) {
        this.offset = 0;
        this.id = id;
        this.pos = pos.clone();
    }

    update(pos: Float32Array, z: Float32Array, u: Float32Array): void {
        let u_i = new Vector3(u[this.offset], u[this.offset + 1], u[this.offset + 2]);
        const p = new Vector3(pos[this.id * 3], pos[this.id * 3 + 1], pos[this.id * 3 + 2]);
        const q = p.add(u_i); // Dix + ui
        const z_i = this.pos;
        u_i = q.subtract(z_i);
        z[this.offset]     = z_i.x;
        z[this.offset + 1] = z_i.y;
        z[this.offset + 2] = z_i.z;
        u[this.offset]     = u_i.x;
        u[this.offset + 1] = u_i.y;
        u[this.offset + 2] = u_i.z;
    }

    getId(): number[] {
        return [this.id];
    }

    getD(): Triplet[] {
        const D = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ];
        const triplets: Triplet[] = [];
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                triplets.push({ row: i, col: j, val: D[i][j] });
            }
        }
        return triplets;
    }

    getW(): number[] {
        const weight = 10000.0;
        return [weight, weight, weight];
    }
}

export class ADMMSolver {
    numVertices: number;
    pos: Float32Array;       // Current vertex positions x
    prevPos: Float32Array;   // Previous vertex positions xₚ
    inertiaPos: Float32Array; // Inertial positions x̄
    vel: Float32Array;       // Velocities v
    masses: Float32Array;    // Masses M
    fixedVertices: Uint8Array;

    numEdges: number;
    edges: Uint32Array;      // Spring connectivity info
    stiffnesses: Float32Array; // Spring stiffness k

    z: Float32Array;         // Local displacement vectors for each spring (m * 3 dimensions)
    u: Float32Array;         // Dual variables for each spring (m * 3 dimensions)

    D: SparseMatrix;         // Reduction matrix
    Dt: SparseMatrix;       // Transpose of the reduction matrix
    W: SparseMatrix;        // Weight matrix
    M: SparseMatrix;       // Mass matrix can be written as a diagonal array but I use a sparse matrix

    Dt_Wt_W: SparseMatrix;   // Precomputed Dt*W*W
    A: SparseMatrix;       // Global system matrix A = M + dt**2*Dt*W*W*D

    params: Params;

    D_rows: number; // Number of rows in the reduction matrix D

    energyTerms: EnergyTerm[];

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
        this.params = params;

        this.D = new SparseMatrix();
        this.Dt = new SparseMatrix();
        this.W = new SparseMatrix();
        this.M = new SparseMatrix();
        this.Dt_Wt_W = new SparseMatrix();
        this.A = new SparseMatrix();

        this.energyTerms = [];

        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            const stiffness = this.stiffnesses[e];
            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);
            const restLength = Vector3.Distance(p0, p1);
            this.energyTerms.push(new SpringEnergyTerm(id0, id1, stiffness, restLength));
        }

        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) {
                const pos = this.getVector3(this.pos, i);
                this.energyTerms.push(new FixedEnergyTerm(i, pos));
            }
        }

        // Construct the reduction matrix D
        const D_triplets: Triplet[] = [];
        const W_triplets: Triplet[] = [];
        let offset = 0;

        for (const term of this.energyTerms) {
            term.offset = offset;
            const localIndices = term.getId();
            const localTriplets = term.getD();
            const weights = term.getW();

            for (const t of localTriplets) {
                const globalNodeIndex = localIndices[Math.floor(t.col / 3)];
                const globalCol = globalNodeIndex * 3 + (t.col % 3);
                D_triplets.push({ row: offset + t.row, col: globalCol, val: t.val });
            }

            for (let i = 0; i < weights.length; i++) {
                W_triplets.push({ row: offset + i, col: offset + i, val: weights[i] });
            }

            offset += weights.length;
        }


        this.D_rows = offset;
        this.D = new SparseMatrix();
        this.D.resize(this.D_rows, this.numVertices * 3);
        this.D.setFromTriplets(D_triplets);
        this.Dt = this.D.transpose();

        this.W = new SparseMatrix();
        this.W.resize(this.D_rows, this.D_rows);
        this.W.setFromTriplets(W_triplets);

        this.z = new Float32Array(this.D_rows);
        this.u = new Float32Array(this.D_rows);

        // Precompute Dt * Wt * W
        const dt = this.params.dt / this.params.numSubsteps;
        const dt2 = dt * dt;
        this.Dt_Wt_W = this.Dt.multiply(this.W).multiply(this.W).scale(dt2);

        // Construct the mass matrix M
        const M_triplets: Triplet[] = [];
        for (let i = 0; i < this.numVertices; i++) {
            const mass = this.masses[i];
            for (let j = 0; j < 3; j++) {
                M_triplets.push({ row: i * 3 + j, col: i * 3 + j, val: mass });
            }
        }
        this.M.resize(this.numVertices * 3, this.numVertices * 3);
        this.M.setFromTriplets(M_triplets);

        this.A = this.M.add(this.Dt_Wt_W.multiply(this.D));
    }

    step(): void {
        for (let step = 0; step < this.params.numSubsteps; step++) {
            this.solve();
        }
    }

    solve(): void {
        const g = this.params.g;
        const dt = this.params.dt / this.params.numSubsteps;
        const invDt = 1 / dt;

        this.prevPos.set(this.pos);

        // Compute inertia positions
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) this.setVector3(this.inertiaPos, i, this.getVector3(this.pos, i));
            else {
                let p = this.getVector3(this.pos, i);
                let v = this.getVector3(this.vel, i);
                v.addInPlace(g.scale(dt)); // Apply gravity
                p.addInPlace(v.scale(dt));
                this.setVector3(this.inertiaPos, i, p);
                this.setVector3(this.pos, i, p);
            }
        }

        // reset
        this.z = this.D.multiplyVector(this.pos);
        this.u.fill(0);

        // ADMM iterations
        for (let itr = 0; itr < this.params.numIterations; itr++) {
            // local step
            for (const term of this.energyTerms) {
                term.update(this.pos, this.z, this.u);
            }

            // global step
            const b = this.M.multiplyVector(this.inertiaPos);

            const z_minus_u = new Float32Array(this.D_rows);
            for (let i = 0; i < this.z.length; i++) {
                z_minus_u[i] = this.z[i] - this.u[i];
            }
            const Dt_Wt_W_z_minus_u = this.Dt_Wt_W.multiplyVector(z_minus_u);
            for (let i = 0; i < this.numVertices * 3; i++) {
                b[i] += Dt_Wt_W_z_minus_u[i];
            }

            const x = this.A.conjugateGradientSolver(b);
            this.pos.set(x);
        }

        for (let i = 0; i < this.numVertices * 3; i++) {
            this.vel[i] = (this.pos[i] - this.prevPos[i]) * invDt;
        }

    }

    getVector3(array: Float32Array, i: number): Vector3 {
        return new Vector3(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
    }

    setVector3(array: Float32Array, i: number, v: Vector3): void {
        array[i * 3] = v.x; array[i * 3 + 1] = v.y; array[i * 3 + 2] = v.z;
    }
}