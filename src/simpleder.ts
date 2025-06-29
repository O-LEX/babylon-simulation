import * as BABYLON from "@babylonjs/core";

// Type alias for readability, exported for use in other modules.
export type Vector = number[];
export type Matrix = number[][];

/**
 * Manages the state and computation for the entire discrete elastic rod system.
 * This class holds all data in flat arrays, representing the whole rod at once.
 */
export class DERSystem {
    // --- Immutable Properties ---
    public readonly numNodes: number;
    public readonly numSegments: number;
    public readonly numHinges: number;
    public readonly dof: number; // Total Degrees of Freedom

    // --- State Variables ---
    public X: Vector; // DOFs: [x0,y0,z0,phi0, x1,y1,z1,phi1, ... x_N-1]

    // --- Intermediate Variables (for the entire rod) ---
    public referenceDirectors: BABYLON.Quaternion[];
    public edges: BABYLON.Vector3[];
    public tangents: BABYLON.Vector3[];
    public directors: BABYLON.Quaternion[];
    public kappas: BABYLON.Vector3[];

    constructor(numNodes: number) {
        this.numNodes = numNodes;
        this.numSegments = numNodes - 1;
        this.numHinges = numNodes - 2;
        this.dof = numNodes * 4 - 1;

        // Initialize state and intermediate variable arrays
        this.X = new Array(this.dof).fill(0);
        this.referenceDirectors = Array.from({ length: this.numSegments }, () => BABYLON.Quaternion.Identity());
        this.edges = Array.from({ length: this.numSegments }, () => BABYLON.Vector3.Zero());
        this.tangents = Array.from({ length: this.numSegments }, () => BABYLON.Vector3.Zero());
        this.directors = Array.from({ length: this.numSegments }, () => BABYLON.Quaternion.Identity());
        this.kappas = Array.from({ length: this.numHinges }, () => BABYLON.Vector3.Zero());
    }

    /**
     * Updates all intermediate physical quantities for the entire rod based on the current state X.
     * This method should be called once per frame/step before any gradient/force calculations.
     */
    public computeAllIntermediateVariables(): void {
        const e3 = new BABYLON.Vector3(0, 0, 1);

        // 1. Compute edges and tangents for all segments
        for (let i = 0; i < this.numSegments; i++) {
            const node0_idx = i * 4;
            const node1_idx = (i + 1) * 4;
            const node0 = new BABYLON.Vector3(this.X[node0_idx], this.X[node0_idx + 1], this.X[node0_idx + 2]);
            const node1 = new BABYLON.Vector3(this.X[node1_idx], this.X[node1_idx + 1], this.X[node1_idx + 2]);
            
            this.edges[i] = node1.subtract(node0);
            this.tangents[i] = this.edges[i].normalizeToNew();
        }

        // 2. Compute directors for all segments
        for (let i = 0; i < this.numSegments; i++) {
            const T_i = e3.applyRotationQuaternion(this.referenceDirectors[i]);
            const t_i = this.tangents[i];
            const phi_i = this.X[i * 4 + 3];

            // Parallel transport quaternion (p_i) using the robust formula from the paper
            const dot = BABYLON.Vector3.Dot(T_i, t_i);
            const p_scalar = Math.sqrt((1 + dot) / 2.0);
            // Avoid division by zero if p_scalar is close to zero
            const p_vec_scale = (p_scalar > 1e-8) ? 1 / (2 * p_scalar) : 0;
            const p_vec = BABYLON.Vector3.Cross(T_i, t_i).scale(p_vec_scale);
            const p_i = new BABYLON.Quaternion(p_vec.x, p_vec.y, p_vec.z, p_scalar);

            // Twist quaternion (r_i)
            const r_i = BABYLON.Quaternion.RotationAxis(T_i, phi_i);

            // Final director (d_i)
            this.directors[i] = p_i.multiply(r_i).multiply(this.referenceDirectors[i]);
        }
        
        // 3. Compute kappas for all hinges
        for (let i = 0; i < this.numHinges; i++) {
            const d_prev = this.directors[i];
            const d_curr = this.directors[i + 1];
            
            const q = d_prev.conjugate().multiply(d_curr);
            this.kappas[i] = new BABYLON.Vector3(q.x, q.y, q.z).scale(2);
        }
    }
}

/**
 * Computes the full Jacobian matrix dK/dX for the entire rod.
 * This is the main computational function to get the deformation gradient.
 * @param system The DERSystem object containing the rod's current state.
 * @returns The (3 * numHinges) x (dof) Jacobian matrix.
 */
export function computeFullJacobianDKdX(system: DERSystem): Matrix {
    const jacobian: Matrix = Array.from({ length: system.numHinges * 3 }, () => Array(system.dof).fill(0));

    // Loop through each hinge to compute its local gradient contribution
    for (let hingeIdx = 0; hingeIdx < system.numHinges; hingeIdx++) {
        const seg0_idx = hingeIdx;
        const seg1_idx = hingeIdx + 1;

        // --- Get pre-computed values for the relevant hinge ---
        const d0 = system.directors[seg0_idx];
        const t0 = system.tangents[seg0_idx];
        const t1 = system.tangents[seg1_idx];
        const T0 = new BABYLON.Vector3(0,0,1).applyRotationQuaternion(system.referenceDirectors[seg0_idx]);
        const T1 = new BABYLON.Vector3(0,0,1).applyRotationQuaternion(system.referenceDirectors[seg1_idx]);
        
        const q_vec = system.kappas[hingeIdx].scale(0.5); // This is Im(q)
        const q_w_sq = 1 - q_vec.lengthSquared();
        const q_w = q_w_sq > 0 ? Math.sqrt(q_w_sq) : 0;
        const q_full = new BABYLON.Quaternion(q_vec.x, q_vec.y, q_vec.z, q_w);

        const len_edge0 = system.edges[seg0_idx].length();
        const len_edge1 = system.edges[seg1_idx].length();

        // Compute local 3x11 Jacobian for this hinge
        for (let j = 0; j < 11; j++) {
            const delta_X_local = new Array(11).fill(0);
            delta_X_local[j] = 1.0;

            const delta_n0 = new BABYLON.Vector3(delta_X_local[0], delta_X_local[1], delta_X_local[2]);
            const delta_phi0 = delta_X_local[3];
            const delta_n1 = new BABYLON.Vector3(delta_X_local[4], delta_X_local[5], delta_X_local[6]);
            const delta_phi1 = delta_X_local[7];
            const delta_n2 = new BABYLON.Vector3(delta_X_local[8], delta_X_local[9], delta_X_local[10]);

            // --- Chain rule calculation (same logic as before, but on pre-computed state) ---
            const delta_e0 = delta_n1.subtract(delta_n0);
            const delta_e1 = delta_n2.subtract(delta_n1);

            const proj_t0 = BABYLON.Matrix.Identity().subtract(BABYLON.Matrix.FromValues(t0.x*t0.x, t0.x*t0.y, t0.x*t0.z, 0, t0.y*t0.x, t0.y*t0.y, t0.y*t0.z, 0, t0.z*t0.x, t0.z*t0.y, t0.z*t0.z, 0, 0,0,0,0));
            const delta_t0 = BABYLON.Vector3.TransformCoordinates(delta_e0, proj_t0).scale(1 / len_edge0);
            
            const proj_t1 = BABYLON.Matrix.Identity().subtract(BABYLON.Matrix.FromValues(t1.x*t1.x, t1.x*t1.y, t1.x*t1.z, 0, t1.y*t1.x, t1.y*t1.y, t1.y*t1.z, 0, t1.z*t1.x, t1.z*t1.y, t1.z*t1.z, 0, 0,0,0,0));
            const delta_t1 = BABYLON.Vector3.TransformCoordinates(delta_e1, proj_t1).scale(1 / len_edge1);

            const k0 = BABYLON.Vector3.Cross(T0, t0).scale(2 / (1 + BABYLON.Vector3.Dot(T0, t0)));
            const k1 = BABYLON.Vector3.Cross(T1, t1).scale(2 / (1 + BABYLON.Vector3.Dot(T1, t1)));

            const delta_p_hat0 = BABYLON.Vector3.Cross(t0, delta_t0).subtract(t0.scale(0.5 * BABYLON.Vector3.Dot(k0, delta_t0)));
            const delta_p_hat1 = BABYLON.Vector3.Cross(t1, delta_t1).subtract(t1.scale(0.5 * BABYLON.Vector3.Dot(k1, delta_t1)));
            
            const delta_d_hat0 = t0.scale(delta_phi0).add(delta_p_hat0);
            const delta_d_hat1 = t1.scale(delta_phi1).add(delta_p_hat1);

            const delta_d_hat_diff = delta_d_hat1.subtract(delta_d_hat0);
            const delta_q_hat = delta_d_hat_diff.applyRotationQuaternion(d0.conjugate());

            const w = q_full.w;
            const v = new BABYLON.Vector3(q_full.x, q_full.y, q_full.z);
            const delta_K = delta_q_hat.scale(w).add(BABYLON.Vector3.Cross(delta_q_hat, v));

            // --- Place local gradient into the global Jacobian ---
            // A local DOF index j corresponds to a global DOF index (hingeIdx*4 + j)
            const global_col_idx = hingeIdx * 4 + j;
            if (global_col_idx < system.dof) {
                jacobian[hingeIdx * 3 + 0][global_col_idx] = delta_K.x;
                jacobian[hingeIdx * 3 + 1][global_col_idx] = delta_K.y;
                jacobian[hingeIdx * 3 + 2][global_col_idx] = delta_K.z;
            }
        }
    }
    return jacobian;
}

// ==============================================================================
// Self-Executing Test Function (runs upon import)
// ==============================================================================

/**
 * Runs a verification test by comparing the analytical Jacobian against a numerical one.
 */
function runJacobianVerificationTest(): void {
    console.log("--- DER Gradient Verification Test ---");

    const NUM_NODES = 4; // Test with 4 nodes -> 3 segments -> 2 hinges
    const system = new DERSystem(NUM_NODES);
    
    // Initialize system with a random state
    system.X = Array.from({ length: system.dof }, () => Math.random());
    
    // Update all intermediate variables needed for the gradient calculation
    system.computeAllIntermediateVariables();
    
    // Compute the Jacobian using the analytical method
    const jacobian_analytic = computeFullJacobianDKdX(system);

    // --- Verification using Finite Difference Method (FDM) ---
    const jacobian_fdm: Matrix = Array.from({ length: system.numHinges * 3 }, () => Array(system.dof).fill(0));
    const eps = 1e-7;
    const baseKappas = system.kappas.flatMap(k => [k.x, k.y, k.z]);

    for (let i = 0; i < system.dof; i++) {
        // Create a temporary system and apply a small perturbation
        const tempSystem = new DERSystem(NUM_NODES);
        tempSystem.X = [...system.X];
        tempSystem.X[i] += eps;
        tempSystem.computeAllIntermediateVariables();
        const newKappas = tempSystem.kappas.flatMap(k => [k.x, k.y, k.z]);
        
        for (let k = 0; k < newKappas.length; k++) {
            jacobian_fdm[k][i] = (newKappas[k] - baseKappas[k]) / eps;
        }
    }
    
    // --- Compare the results ---
    let error = 0;
    for(let i=0; i<jacobian_analytic.length; i++) {
        for(let j=0; j<jacobian_analytic[i].length; j++) {
            error += (jacobian_analytic[i][j] - jacobian_fdm[i][j])**2;
        }
    }
    error = Math.sqrt(error);

    console.log(`\n[Verification Result]`);
    console.log(`Norm of difference between analytical and FDM Jacobians: ${error.toExponential(2)}`);
    if (error < 1e-5) {
        console.log("✅ Test PASSED: The analytical gradient calculation appears to be correct.");
    } else {
        console.log("❌ Test FAILED: Please check the implementation.");
    }
    console.log("--- Verification Test Finished ---");
}

// This line calls the test function. It will execute when this module is imported.
runJacobianVerificationTest();