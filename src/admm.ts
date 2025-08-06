import { Vector3 } from "@babylonjs/core";
import { Triplet, SparseMatrix} from "./util";
import { Geometry } from "./geometry";
import { Params } from "./params";

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
    restLengths: Float32Array; // Spring rest lengths L₀

    z: Float32Array;         // Local displacement vectors for each spring (m * 3 dimensions)
    u: Float32Array;         // Dual variables for each spring (m * 3 dimensions)

    D: SparseMatrix;         // Reduction matrix
    Dt: SparseMatrix;       // Transpose of the reduction matrix
    W: SparseMatrix;        // Weight matrix

    A: SparseMatrix;       // Global system matrix A = M/dt² + Dt*W*W*D
    Dt_Wt_W: SparseMatrix;   // Precomputed Dt*W*W

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

        this.z = new Float32Array(this.numEdges * 3);
        this.u = new Float32Array(this.numEdges * 3);

        this.D = new SparseMatrix();
        this.Dt = new SparseMatrix();
        this.W = new SparseMatrix();

        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            
            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);

            this.restLengths[e] = Vector3.Distance(p0, p1);
        }

        const D_triplets: Triplet[] = [];
        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            const stiffness = this.stiffnesses[e];
            const D_e = [[-1, 0, 0, 1, 0, 0], [0, -1, 0, 0, 1, 0], [0, 0, -1, 0, 0, 1]];
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++) {
                    D_triplets.push({ row: e * 3 + i, col: id0 * 3 + j, val: D_e[i][j] });
                    D_triplets.push({ row: e * 3 + i, col: id1 * 3 + j, val: D_e[i][j + 3] });
                }
            }
        }

        this.D.resize(this.numEdges * 3, this.numVertices * 3);
        this.D.setFromTriplets(D_triplets);
        this.Dt = this.D.transpose();

        const W_triplets: Triplet[] = [];
        for (let e = 0; e < this.numEdges; e++) {
            const weight = Math.sqrt(this.stiffnesses[e]);
            for (let i = 0; i < 3; i++) {
                W_triplets.push({ row: e * 3 + i, col: e * 3 + i, val: weight });
            }
        }

        this.W.resize(this.numEdges * 3, this.numEdges * 3);
        this.W.setFromTriplets(W_triplets);

        const dt = this.params.dt / this.params.numSubsteps;
        const dt2 = dt * dt;
        this.Dt_Wt_W = this.Dt.multiply(this.W).multiply(this.W).scale(dt2);

        // mass matrix M = diag(masses)
        const M_triplets: Triplet[] = [];
        for (let i = 0; i < this.numVertices; i++) {
            const mass = this.masses[i];
            for (let j = 0; j < 3; j++) {
                M_triplets.push({ row: i * 3 + j, col: i * 3 + j, val: mass });
            }
        }
        const M = new SparseMatrix();
        M.resize(this.numVertices * 3, this.numVertices * 3);
        M.setFromTriplets(M_triplets);
        this.A = M.add(this.Dt_Wt_W.multiply(this.D));
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
        this.z = this.D.multiplyVector(this.pos);
        this.u.fill(0);
        const b = new Float32Array(this.numVertices * 3);

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

        // ADMM iterations
        for (let itr = 0; itr < this.params.numIterations; itr++) {
            // local step
            for (let e = 0; e < this.numEdges; e++) {
                const id0 = this.edges[e * 2];
                const id1 = this.edges[e * 2 + 1];
                const p0 = this.getVector3(this.pos, id0);
                const p1 = this.getVector3(this.pos, id1);
                const restLength = this.restLengths[e];
                let z_e = this.getVector3(this.z, e);
                let u_e = this.getVector3(this.u, e);

                const Dix = p1.subtract(p0);
                const q = Dix.add(u_e);

                let p = new Vector3(0, 0, 0);
                const q_length = q.length();
                if (q_length > 1e-9) {
                    p = q.scale(restLength / q_length);
                }

                z_e = p.add(q).scale(0.5);
                u_e = u_e.add(Dix).subtract(z_e);

                this.setVector3(this.z, e, z_e);
                this.setVector3(this.u, e, u_e);
            }

            // global step
            for (let i = 0; i < this.numVertices * 3; i++) {
                b[i] = this.masses[Math.floor(i / 3)] * this.inertiaPos[i];
            }
            
            const z_minus_u = new Float32Array(this.numEdges * 3);
            for (let i = 0; i < this.z.length; i++) {
                z_minus_u[i] = this.z[i] - this.u[i];
            }
            const Dt_Wt_W_z_minus_u = this.Dt_Wt_W.multiplyVector(z_minus_u);
            for (let i = 0; i < this.numVertices * 3; i++) {
                b[i] += Dt_Wt_W_z_minus_u[i];
            }

            const x = this.A.conjugateGradientSolver(b);
            for(let i=0; i<this.numVertices; ++i){
                if(this.fixedVertices[i]){
                    this.setVector3(x, i, this.getVector3(this.inertiaPos, i));
                }
            }
            this.pos.set(x);
        }

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