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

class IPCEnergyTerm implements EnergyTerm {
    offset: number;
    private id: number;
    private id_t0: number;
    private id_t1: number;
    private id_t2: number;
    private stiffness: number;
    private d_hat: number;

    private readonly GRAD_DESCENT_STEPS = 5;
    private readonly LEARNING_RATE = 0.5;

    constructor(id: number, id_t0: number, id_t1: number, id_t2: number, stiffness: number, d_hat: number) {
        this.offset = 0;
        this.id = id;
        this.id_t0 = id_t0;
        this.id_t1 = id_t1;
        this.id_t2 = id_t2;
        this.stiffness = stiffness;
        this.d_hat = d_hat;
    }

    update(pos: Float32Array, z: Float32Array, u: Float32Array): void {
        const p   = new Vector3(pos[this.id * 3], pos[this.id * 3 + 1], pos[this.id * 3 + 2]);
        const t0  = new Vector3(pos[this.id_t0 * 3], pos[this.id_t0 * 3 + 1], pos[this.id_t0 * 3 + 2]);
        const t1  = new Vector3(pos[this.id_t1 * 3], pos[this.id_t1 * 3 + 1], pos[this.id_t1 * 3 + 2]);
        const t2  = new Vector3(pos[this.id_t2 * 3], pos[this.id_t2 * 3 + 1], pos[this.id_t2 * 3 + 2]);
        
        const u_p  = new Vector3(u[this.offset + 0], u[this.offset + 1], u[this.offset + 2]);
        const u_t0 = new Vector3(u[this.offset + 3], u[this.offset + 4], u[this.offset + 5]);
        const u_t1 = new Vector3(u[this.offset + 6], u[this.offset + 7], u[this.offset + 8]);
        const u_t2 = new Vector3(u[this.offset + 9], u[this.offset + 10], u[this.offset + 11]);

        const q_p = p.add(u_p); // Dix + ui
        const q_t0 = t0.add(u_t0);
        const q_t1 = t1.add(u_t1);
        const q_t2 = t2.add(u_t2);
        
        let z_p = q_p.clone();
        let z_t0 = q_t0.clone();
        let z_t1 = q_t1.clone();
        let z_t2 = q_t2.clone();

        const { distance } = this.pointTriangleDistance(z_p, z_t0, z_t1, z_t2);

        // 4. 活性化距離より遠い場合は何もしない (U_ipc = 0)
        // このとき、arg min の解は z = q となる
        if (distance >= this.d_hat || distance <= 1e-9) {
            // zはqのまま
        } else {
            // 5. 勾配降下法で z を更新
            const w_sq = this.stiffness; // w^2 = k

            for (let i = 0; i < this.GRAD_DESCENT_STEPS; i++) {
                // 現状のzでのIPCエネルギーの勾配を計算
                const { grad_d_p, grad_d_t0, grad_d_t1, grad_d_t2 } = this.gradIpcEnergy(z_p, z_t0, z_t1, z_t2);

                // 全体の目的関数の勾配: ∇U_ipc(z) - w^2 * (q - z)
                const grad_f_p = grad_d_p.subtract(q_p.subtract(z_p).scale(w_sq));
                const grad_f_t0 = grad_d_t0.subtract(q_t0.subtract(z_t0).scale(w_sq));
                const grad_f_t1 = grad_d_t1.subtract(q_t1.subtract(z_t1).scale(w_sq));
                const grad_f_t2 = grad_d_t2.subtract(q_t2.subtract(z_t2).scale(w_sq));
                
                // zを更新
                z_p.subtractInPlace(grad_f_p.scale(this.LEARNING_RATE));
                z_t0.subtractInPlace(grad_f_t0.scale(this.LEARNING_RATE));
                z_t1.subtractInPlace(grad_f_t1.scale(this.LEARNING_RATE));
                z_t2.subtractInPlace(grad_f_t2.scale(this.LEARNING_RATE));
            }
        }
        
        // 6. 新しい u_i = q - z_i を計算
        const new_u_p  = q_p.subtract(z_p);
        const new_u_t0 = q_t0.subtract(z_t0);
        const new_u_t1 = q_t1.subtract(z_t1);
        const new_u_t2 = q_t2.subtract(z_t2);

        // 7. グローバル配列 z, u を更新
        [z_p, z_t0, z_t1, z_t2].forEach((v, i) => {
            z[this.offset + i * 3 + 0] = v.x;
            z[this.offset + i * 3 + 1] = v.y;
            z[this.offset + i * 3 + 2] = v.z;
        });
        [new_u_p, new_u_t0, new_u_t1, new_u_t2].forEach((v, i) => {
            u[this.offset + i * 3 + 0] = v.x;
            u[this.offset + i * 3 + 1] = v.y;
            u[this.offset + i * 3 + 2] = v.z;
        });
    }

    /**
     * IPCエネルギーとその勾配を計算します。
     * E(d) = -k * (d - d_hat)^2 * log(d / d_hat)
     * ∇E = (dE/dd) * ∇d
     */
    private gradIpcEnergy(p: Vector3, t0: Vector3, t1: Vector3, t2: Vector3) {
        const { distance, grad_d_p, grad_d_t0, grad_d_t1, grad_d_t2 } = this.pointTriangleDistance(p, t0, t1, t2);

        if (distance >= this.d_hat || distance <= 1e-9) {
             return { 
                grad_d_p: Vector3.Zero(), grad_d_t0: Vector3.Zero(), 
                grad_d_t1: Vector3.Zero(), grad_d_t2: Vector3.Zero() 
            };
        }
        
        // dE/dd の計算
        const term1 = -2 * this.stiffness * (distance - this.d_hat) * Math.log(distance / this.d_hat);
        const term2 = -this.stiffness * Math.pow(distance - this.d_hat, 2) / distance;
        const dE_dd = term1 + term2;
        
        return {
            grad_d_p: grad_d_p.scale(dE_dd),
            grad_d_t0: grad_d_t0.scale(dE_dd),
            grad_d_t1: grad_d_t1.scale(dE_dd),
            grad_d_t2: grad_d_t2.scale(dE_dd),
        };
    }

    /**
     * 頂点-三角形の距離と勾配を計算します。（簡略版）
     * この実装は頂点が面に射影されるケースのみを考慮しています。
     * 厳密なIPCには、辺や頂点への最近傍点の考慮が必要です。
     */
    private pointTriangleDistance(p: Vector3, t0: Vector3, t1: Vector3, t2: Vector3) {
        const t1_t0 = t1.subtract(t0);
        const t2_t0 = t2.subtract(t0);
        const p_t0 = p.subtract(t0);
        
        const normal = Vector3.Cross(t1_t0, t2_t0);
        normal.normalize();

        const distance = Vector3.Dot(p_t0, normal);
        
        // 勾配 ∇d の計算 (d = n · (p - t0))
        // ∇p d = n
        // ∇t0 d = (b-1)n
        // ∇t1 d = (c-1)n
        // ∇t2 d = - (b+c-1)n  これは間違い -> ∇t0 d = -n, ∇ti dは複雑
        // 簡略化のため、pに対する勾配のみを正確に扱い、三角形側は分配します。
        // これは正確な勾配ではありませんが、反発力としては機能します。
        const grad_d_p = normal;
        const grad_d_t0 = normal.scale(-0.33);
        const grad_d_t1 = normal.scale(-0.33);
        const grad_d_t2 = normal.scale(-0.33);

        return { distance, grad_d_p, grad_d_t0, grad_d_t1, grad_d_t2 };
    }


    /**
     * 関与する頂点のIDリストを返します。
     */
    getId(): number[] {
        return [this.id, this.id_t0, this.id_t1, this.id_t2];
    }

    /**
     * リダクション行列Dを返します。
     * この項では4つの頂点（12自由度）を扱うため、12x12の単位行列に対応します。
     */
    getD(): Triplet[] {
        const triplets: Triplet[] = [];
        for (let i = 0; i < 12; i++) {
            triplets.push({ row: i, col: i, val: 1 });
        }
        return triplets;
    }

    getW(): number[] {
        const weight = Math.sqrt(this.stiffness);
        return Array(12).fill(weight);
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

    triangles?: Uint32Array;     // Triangle connectivity info

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

        // ipc
        if (geometry.triangles) {
            this.triangles = new Uint32Array(geometry.triangles);
            for (let i = 0; i < this.numVertices; i++) {
                const id = i;
                for (let j = 0; j < this.triangles.length; j += 3) {
                    const id0 = this.triangles[j];
                    const id1 = this.triangles[j + 1];
                    const id2 = this.triangles[j + 2];
                    this.energyTerms.push(new IPCEnergyTerm(id, id0, id1, id2, 1000, 0.5));
                }
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