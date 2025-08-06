import { Vector3 } from "@babylonjs/core";
import { Matrix3x3, BlockSparseMatrix, SparseMatrix } from "./util";
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

    zs: Float32Array;         // Local displacement vectors for each spring (m * 3 dimensions)
    us: Float32Array;         // Dual variables for each spring (m * 3 dimensions)

    // Matrices for the global step (computed only once)
    A: BlockSparseMatrix;    // M/dt² + Dt*W*W*D
    D: SparseMatrix;         // Reduction matrix
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

        this.zs = new Float32Array(this.numEdges * 3);
        this.us = new Float32Array(this.numEdges * 3);

        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];
            
            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);

            this.restLengths[e] = Vector3.Distance(p0, p1);
        }

        this.D = new SparseMatrix();

        const structure = this.createSparseMatrixStructure();
        this.D.initialize(structure.row2col, structure.idx2col);
    }

    private createSparseMatrixStructure(): { row2col: number[], idx2col: number[] } {
        const row2col: number[] = [];
        const idx2col: number[] = [];

        for (let i = 0; i < this.numVertices; i++) {
            row2col.push(i);
            idx2col.push(i * 3, i * 3 + 1, i * 3 + 2);
        }

        return { row2col, idx2col };
    }

    private initializeSystem(): void {
        for 

        // 1. Create D and W^2 matrices
        // D is a matrix indicating which vertices (i,j) each spring (edge e) connects,
        // such that D_e * x = x_j - x_i.
        // W contains the weights; following the paper, we use the sqrt of stiffness.
        // Here, we would construct the matrix Dt*W*W.

        // 2. Compute the global matrix A = M/dt² + Dt*W*W*D
        // This matrix A is constant throughout the simulation.
        // For efficiency, one would perform Cholesky decomposition here.
    }

    step(): void {
        for (let step = 0; step < this.params.numSubsteps; step++) {
            this.solve();
        }
    }

    solve(): void {
        this.prevPos.set(this.pos);
        this.us.fill(0);

        const g = this.params.g;
        const dt = this.params.dt / this.params.numSubsteps;
        const invDt = 1 / dt;
        const invDt2 = 1 / (dt * dt);

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
                const stiffness = this.stiffnesses[e];
                const z = this.getVector3(this.zs, e);
                const u = this.getVector3(this.us, e);

                // Compute D_e * x
                const Dix = p1.subtract(p0);
                
                // Compute the target vector d = Dᵢx + uᵢ
                const d = Dix.add(ue);

                // Update z_e: project d onto the rest length L₀
                const ze = d.normalize().scale(restLength);
                this.setVector3(this.z, e, ze);

                // Update u_e: u_e = u_e + (Dᵢx - z_e)
                const new_ue = ue.add(Dix.subtract(ze));
                this.setVector3(this.u, e, new_ue);
            }

            // ==============
            //  Global Step
            // ==============
            // Form the right-hand side vector b = (M/dt²) * x_bar + Dt*W*W*(z-u)
            const b = new Float32Array(this.numVertices * 3);
            
            // The (M/dt²) * x_bar part
            for(let i=0; i<this.numVertices; ++i){
                const x_bar_i = this.getVector3(x_bar, i);
                const b_i = x_bar_i.scale(this.masses[i] * invDt2);
                this.addVector3(b, i, b_i); // Add to b
            }

            // The Dt*W*W*(z-u) part
            // Compute (z-u), multiply by Dt*W*W, and add to the corresponding vertices.
            // ... (implementation details) ...

            // Handle fixed vertices
            // ...
            
            // Solve the linear system A * pos = b
            // A is constant, so we use the one computed/factorized during initialization.
            const new_pos_flat = this.A.conjugateGradientSolver(b, 100, 1e-6);
            this.pos.set(new_pos_flat);
        }

        // 3. Update velocities
        // ... (similar to ImplicitSolver)
    }

    getVector3(array: Float32Array, i: number): Vector3 {
        return new Vector3(array[i * 3], array[i * 3 + 1], array[i * 3 + 2]);
    }

    setVector3(array: Float32Array, i: number, v: Vector3): void {
        array[i * 3] = v.x; array[i * 3 + 1] = v.y; array[i * 3 + 2] = v.z;
    }
}