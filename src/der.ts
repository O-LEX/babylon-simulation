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

// Discrete Elastic Rod geometry with alternating position-theta structure
export class DERGeometry {
    q: Vector3[]; // Alternating vector: q[2*i] = position[i], q[2*i+1] = theta[i] (x component only)
    initialQ: Vector3[];
    
    // Element data stored as Structure of Arrays for better cache efficiency
    restLengths: number[] = [];        // Rest length for each edge
    restTwists: number[] = [];         // Rest twist for each edge  
    restCurvatures: Vector3[] = [];    // Rest curvature for each edge
    
    fixedBlocks: Set<number>; // Fixed blocks in q vector
    
    // Material properties
    stretchStiffness: number;
    bendStiffness: number;
    twistStiffness: number;
    vertexMass: number;
    radius: number; // Rod radius

    constructor(
        positions: Vector3[], 
        thetas: number[],
        stretchStiffness: number = 1000,
        bendStiffness: number = 100,
        twistStiffness: number = 100,
        vertexMass: number = 1.0,
        radius: number = 0.1
    ) {
        // Create alternating q vector: position, theta, position, theta, ...
        this.q = [];
        this.initialQ = [];
        
        for (let i = 0; i < positions.length; i++) {
            // Position block
            this.q.push(positions[i].clone());
            this.initialQ.push(positions[i].clone());
            
            // Theta block (only x component is used for twist)
            this.q.push(new Vector3(thetas[i], 0, 0));
            this.initialQ.push(new Vector3(thetas[i], 0, 0));
        }
        
        this.fixedBlocks = new Set();
        this.stretchStiffness = stretchStiffness;
        this.bendStiffness = bendStiffness;
        this.twistStiffness = twistStiffness;
        this.vertexMass = vertexMass;
        this.radius = radius;
        
        this.createElementData(positions);
    }

    private createElementData(positions: Vector3[]): void {
        const numEdges = positions.length - 1;
        this.restLengths = new Array(numEdges);
        this.restTwists = new Array(numEdges);
        this.restCurvatures = new Array(numEdges);
        
        for (let i = 0; i < numEdges; i++) {
            this.restLengths[i] = Vector3.Distance(positions[i], positions[i + 1]);
            this.restTwists[i] = 0; // Initial twist
            this.restCurvatures[i] = new Vector3(0, 0, 0); // Initial curvature
        }
    }

    // Convert vertex index to position block index in q
    getPositionBlockIndex(vertexIndex: number): number {
        return 2 * vertexIndex;
    }

    // Convert vertex index to theta block index in q
    getThetaBlockIndex(vertexIndex: number): number {
        return 2 * vertexIndex + 1;
    }

    // Get position from q vector
    getPosition(vertexIndex: number): Vector3 {
        return this.q[this.getPositionBlockIndex(vertexIndex)];
    }

    // Get theta value from q vector
    getTheta(vertexIndex: number): number {
        return this.q[this.getThetaBlockIndex(vertexIndex)].x;
    }

    // Set position in q vector
    setPosition(vertexIndex: number, position: Vector3): void {
        this.q[this.getPositionBlockIndex(vertexIndex)] = position.clone();
    }

    // Set theta in q vector
    setTheta(vertexIndex: number, theta: number): void {
        this.q[this.getThetaBlockIndex(vertexIndex)] = new Vector3(theta, 0, 0);
    }

    setFixedVertex(vertexIndex: number): void {
        this.fixedBlocks.add(this.getPositionBlockIndex(vertexIndex));
        this.fixedBlocks.add(this.getThetaBlockIndex(vertexIndex));
    }

    setFixedVertices(vertexIndices: number[]): void {
        for (const index of vertexIndices) {
            this.setFixedVertex(index);
        }
    }

    isBlockFixed(blockIndex: number): boolean {
        return this.fixedBlocks.has(blockIndex);
    }

    isVertexFixed(vertexIndex: number): boolean {
        return this.fixedBlocks.has(this.getPositionBlockIndex(vertexIndex));
    }

    getNumVertices(): number {
        return this.q.length / 2;
    }

    getNumBlocks(): number {
        return this.q.length;
    }

    getNumElements(): number {
        return this.restLengths.length;
    }

    // Get all positions as Vector3 array
    getPositions(): Vector3[] {
        const positions = [];
        for (let i = 0; i < this.getNumVertices(); i++) {
            positions.push(this.getPosition(i));
        }
        return positions;
    }

    // Get all thetas as number array
    getThetas(): number[] {
        const thetas = [];
        for (let i = 0; i < this.getNumVertices(); i++) {
            thetas.push(this.getTheta(i));
        }
        return thetas;
    }
}

// Discrete Elastic Rod solver using discrete viscous thread model
export class DERSolver {
    private geometry: DERGeometry;
    private bsm: BlockSparseMatrix;
    private velocities: Vector3[]; // Velocities for q vector
    private gravity: Vector3 = new Vector3(0, -9.81, 0);

    constructor(geometry: DERGeometry) {
        this.geometry = geometry;
        this.bsm = new BlockSparseMatrix();
        
        // Create sparse matrix structure for q vector
        const structure = this.createSparseMatrixStructure(geometry);
        this.bsm.initialize(structure.row2idx, structure.idx2col);
        
        // Initialize velocities to zero for all blocks in q
        this.velocities = new Array(geometry.getNumBlocks())
            .fill(null).map(() => new Vector3(0, 0, 0));
    }

    private createSparseMatrixStructure(geometry: DERGeometry): { row2idx: number[], idx2col: number[] } {
        const numBlocks = geometry.getNumBlocks();
        const numVertices = geometry.getNumVertices();
        const adjacency = new Map<number, Set<number>>();
        
        // Initialize adjacency lists
        for (let i = 0; i < numBlocks; i++) {
            adjacency.set(i, new Set([i])); // Self-connection
        }
        
        // Add element connections for positions (edges connect consecutive vertices)
        for (let i = 0; i < geometry.getNumElements(); i++) {
            const v0 = i;
            const v1 = i + 1;
            const blockV0 = geometry.getPositionBlockIndex(v0);
            const blockV1 = geometry.getPositionBlockIndex(v1);
            adjacency.get(blockV0)!.add(blockV1);
            adjacency.get(blockV1)!.add(blockV0);
        }
        
        // Add connections for bending (positions affect each other)
        for (let i = 0; i < numVertices - 2; i++) {
            for (let j = i; j <= i + 2; j++) {
                for (let k = i; k <= i + 2; k++) {
                    if (j < numVertices && k < numVertices) {
                        const blockJ = geometry.getPositionBlockIndex(j);
                        const blockK = geometry.getPositionBlockIndex(k);
                        adjacency.get(blockJ)!.add(blockK);
                    }
                }
            }
        }
        
        // Add connections for twisting (positions and thetas affect each other)
        for (let i = 0; i < numVertices - 2; i++) {
            for (let j = i; j <= i + 2; j++) {
                for (let k = i; k <= i + 2; k++) {
                    if (j < numVertices && k < numVertices) {
                        const posJ = geometry.getPositionBlockIndex(j);
                        const posK = geometry.getPositionBlockIndex(k);
                        const thetaJ = geometry.getThetaBlockIndex(j);
                        const thetaK = geometry.getThetaBlockIndex(k);
                        
                        // Position-position connections
                        adjacency.get(posJ)!.add(posK);
                        // Position-theta connections
                        adjacency.get(posJ)!.add(thetaK);
                        adjacency.get(thetaJ)!.add(posK);
                        // Theta-theta connections
                        adjacency.get(thetaJ)!.add(thetaK);
                    }
                }
            }
        }
        
        // Build CSR structure
        const row2idx: number[] = [0];
        const idx2col: number[] = [];
        
        for (let i = 0; i < numBlocks; i++) {
            const neighbors = Array.from(adjacency.get(i)!).sort((a, b) => a - b);
            for (const neighbor of neighbors) {
                idx2col.push(neighbor);
            }
            row2idx.push(idx2col.length);
        }
        
        return { row2idx, idx2col };
    }

    // Stretching energy computation
    private computeStretchingForces(
        p0: Vector3, 
        p1: Vector3, 
        restLength: number,
        stiffness: number
    ): { energy: number, gradients: Vector3[], hessian: Matrix3x3[][] } {
        const edge = p1.subtract(p0);
        const currentLength = edge.length();
        
        if (currentLength < 1e-8) {
            return {
                energy: 0,
                gradients: [new Vector3(0, 0, 0), new Vector3(0, 0, 0)],
                hessian: [[Matrix3x3.identity().scale(stiffness), Matrix3x3.identity().scale(-stiffness)], 
                         [Matrix3x3.identity().scale(-stiffness), Matrix3x3.identity().scale(stiffness)]]
            };
        }
        
        const strain = currentLength - restLength;
        const energy = 0.5 * stiffness * strain * strain;
        
        const u = edge.normalize();
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

    // Bending energy computation using discrete curvature
    private computeBendingForces(
        p0: Vector3, 
        p1: Vector3, 
        p2: Vector3,
        stiffness: number
    ): { energy: number, gradients: Vector3[], hessian: Matrix3x3[][] } {
        const e1 = p1.subtract(p0);
        const e2 = p2.subtract(p1);
        
        const l1 = e1.length();
        const l2 = e2.length();
        
        if (l1 < 1e-8 || l2 < 1e-8) {
            return {
                energy: 0,
                gradients: [new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)],
                hessian: [
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()]
                ]
            };
        }
        
        // Normalized tangent vectors
        const t1 = e1.scale(1.0 / l1);
        const t2 = e2.scale(1.0 / l2);
        
        // Discrete curvature vector: κ = 2 * (t2 - t1) / (l1 + l2)
        const avgLength = (l1 + l2) * 0.5;
        const kappa = t2.subtract(t1).scale(2.0 / (l1 + l2));
        const kappaSquared = Vector3.Dot(kappa, kappa);
        
        // Bending energy: E = (1/2) * EI * |κ|^2 * avgLength
        const energy = 0.5 * stiffness * kappaSquared * avgLength;
        
        // Compute gradients analytically
        const factor = stiffness * avgLength * 2.0 / (l1 + l2);
        
        // ∂κ/∂p0 = -2/(l1+l2) * (1/l1 * I - t1⊗t1/l1)
        const t1OuterT1 = Matrix3x3.outerProduct(t1, t1);
        const dKappa_dp0_matrix = Matrix3x3.identity().subtract(t1OuterT1).scale(-2.0 / ((l1 + l2) * l1));
        const grad0 = dKappa_dp0_matrix.multiplyVector(kappa).scale(factor);
        
        // ∂κ/∂p2 = 2/(l1+l2) * (1/l2 * I - t2⊗t2/l2)  
        const t2OuterT2 = Matrix3x3.outerProduct(t2, t2);
        const dKappa_dp2_matrix = Matrix3x3.identity().subtract(t2OuterT2).scale(2.0 / ((l1 + l2) * l2));
        const grad2 = dKappa_dp2_matrix.multiplyVector(kappa).scale(factor);
        
        // ∂κ/∂p1 = -(∂κ/∂p0 + ∂κ/∂p2)
        const grad1 = grad0.add(grad2).scale(-1);
        
        const gradients = [grad0, grad1, grad2];
        
        // Compute Hessian matrices
        const hessianFactor = stiffness * avgLength * 4.0 / ((l1 + l2) * (l1 + l2));
        
        // Simplified Hessian computation for stability and correctness
        const baseHessian = Matrix3x3.identity().scale(stiffness * 0.01); // Small regularization
        const gradientContrib = Matrix3x3.outerProduct(kappa, kappa).scale(hessianFactor);
        
        const H00 = baseHessian.add(gradientContrib.scale(1.0 / (l1 * l1)));
        const H22 = baseHessian.add(gradientContrib.scale(1.0 / (l2 * l2)));
        const H01 = gradientContrib.scale(-1.0 / l1);
        const H12 = gradientContrib.scale(-1.0 / l2);
        const H02 = new Matrix3x3(); // Zero for non-adjacent vertices
        const H11 = H00.add(H22).add(H01.scale(2)).add(H12.scale(2));
        
        const hessian = [
            [H00, H01, H02],
            [H01, H11, H12],
            [H02, H12, H22]
        ];
        
        return { energy, gradients, hessian };
    }

    // Twisting energy computation using discrete viscous thread model
    private computeTwistingForces(
        p0: Vector3,
        p1: Vector3,
        p2: Vector3,
        theta0: number,
        theta1: number,
        theta2: number,
        stiffness: number
    ): { 
        energy: number, 
        positionGradients: Vector3[], 
        thetaGradients: Vector3[],
        positionHessian: Matrix3x3[][],
        thetaHessian: Matrix3x3[][],
        crossHessian: Matrix3x3[][]
    } {
        const e1 = p1.subtract(p0);
        const e2 = p2.subtract(p1);
        
        const l1 = e1.length();
        const l2 = e2.length();
        
        if (l1 < 1e-8 || l2 < 1e-8) {
            return {
                energy: 0,
                positionGradients: [new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)],
                thetaGradients: [new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)],
                positionHessian: [
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()]
                ],
                thetaHessian: [
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()]
                ],
                crossHessian: [
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
                    [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()]
                ]
            };
        }
        
        // Discrete twist computation
        const avgLength = (l1 + l2) * 0.5;
        const twist = (theta2 - theta1) / avgLength - (theta1 - theta0) / avgLength;
        
        const energy = 0.5 * stiffness * twist * twist;
        
        // Gradients (twist affects both positions and theta values)
        const factor = stiffness * twist / avgLength;
        
        // Position gradients (simplified - twist primarily affects theta)
        const positionGradients = [
            new Vector3(0, 0, 0), // Simplified - twist primarily affects theta
            new Vector3(0, 0, 0),
            new Vector3(0, 0, 0)
        ];
        
        // Theta gradients (only x component is used)
        const thetaGradients = [
            new Vector3(factor * (-1.0 / avgLength), 0, 0),
            new Vector3(factor * (2.0 / avgLength), 0, 0),
            new Vector3(factor * (-1.0 / avgLength), 0, 0)
        ];
        
        // Simplified hessians
        const thetaH = Matrix3x3.identity().scale(stiffness / (avgLength * avgLength));
        const thetaHessian = [
            [thetaH, thetaH.scale(-1), new Matrix3x3()],
            [thetaH.scale(-1), thetaH.scale(2), thetaH.scale(-1)],
            [new Matrix3x3(), thetaH.scale(-1), thetaH]
        ];
        
        const positionHessian = [
            [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
            [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
            [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()]
        ];
        
        const crossHessian = [
            [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
            [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()],
            [new Matrix3x3(), new Matrix3x3(), new Matrix3x3()]
        ];
        
        return { 
            energy, 
            positionGradients, 
            thetaGradients, 
            positionHessian, 
            thetaHessian, 
            crossHessian 
        };
    }

    step(deltaTime: number): void {
        const numBlocks = this.geometry.getNumBlocks();
        const numVertices = this.geometry.getNumVertices();
        const gradient = new Array(numBlocks).fill(null).map(() => new Vector3(0, 0, 0));
        
        this.bsm.setZero();
        
        // Step 1: Update q using current velocities
        for (let i = 0; i < numBlocks; i++) {
            if (!this.geometry.isBlockFixed(i)) {
                this.geometry.q[i].addInPlace(this.velocities[i].scale(deltaTime));
            }
        }
        
        let totalEnergy = 0;
        
        // Step 2: Process stretching forces
        for (let i = 0; i < this.geometry.getNumElements(); i++) {
            const v0 = i;      // First vertex of edge i
            const v1 = i + 1;  // Second vertex of edge i
            
            const result = this.computeStretchingForces(
                this.geometry.getPosition(v0),
                this.geometry.getPosition(v1),
                this.geometry.restLengths[i],
                this.geometry.stretchStiffness
            );
            
            totalEnergy += result.energy;
            
            const pos0Block = this.geometry.getPositionBlockIndex(v0);
            const pos1Block = this.geometry.getPositionBlockIndex(v1);
            
            gradient[pos0Block].addInPlace(result.gradients[0]);
            gradient[pos1Block].addInPlace(result.gradients[1]);
            
            this.bsm.addBlockAt(pos0Block, pos0Block, result.hessian[0][0]);
            this.bsm.addBlockAt(pos0Block, pos1Block, result.hessian[0][1]);
            this.bsm.addBlockAt(pos1Block, pos0Block, result.hessian[1][0]);
            this.bsm.addBlockAt(pos1Block, pos1Block, result.hessian[1][1]);
        }
        
        // Step 3: Process bending forces
        for (let i = 0; i < numVertices - 2; i++) {
            const result = this.computeBendingForces(
                this.geometry.getPosition(i),
                this.geometry.getPosition(i + 1),
                this.geometry.getPosition(i + 2),
                this.geometry.bendStiffness
            );
            
            totalEnergy += result.energy;
            
            const pos0Block = this.geometry.getPositionBlockIndex(i);
            const pos1Block = this.geometry.getPositionBlockIndex(i + 1);
            const pos2Block = this.geometry.getPositionBlockIndex(i + 2);
            
            gradient[pos0Block].addInPlace(result.gradients[0]);
            gradient[pos1Block].addInPlace(result.gradients[1]);
            gradient[pos2Block].addInPlace(result.gradients[2]);
            
            // Add hessian blocks
            const posBlocks = [pos0Block, pos1Block, pos2Block];
            for (let j = 0; j < 3; j++) {
                for (let k = 0; k < 3; k++) {
                    this.bsm.addBlockAt(posBlocks[j], posBlocks[k], result.hessian[j][k]);
                }
            }
        }
        
        // Step 4: Process twisting forces (DISABLED FOR STRETCHING TEST)
        /*
        for (let i = 0; i < numVertices - 2; i++) {
            const result = this.computeTwistingForces(
                this.geometry.getPosition(i),
                this.geometry.getPosition(i + 1),
                this.geometry.getPosition(i + 2),
                this.geometry.getTheta(i),
                this.geometry.getTheta(i + 1),
                this.geometry.getTheta(i + 2),
                this.geometry.twistStiffness
            );
            
            totalEnergy += result.energy;
            
            const pos0Block = this.geometry.getPositionBlockIndex(i);
            const pos1Block = this.geometry.getPositionBlockIndex(i + 1);
            const pos2Block = this.geometry.getPositionBlockIndex(i + 2);
            const theta0Block = this.geometry.getThetaBlockIndex(i);
            const theta1Block = this.geometry.getThetaBlockIndex(i + 1);
            const theta2Block = this.geometry.getThetaBlockIndex(i + 2);
            
            // Add position gradients
            gradient[pos0Block].addInPlace(result.positionGradients[0]);
            gradient[pos1Block].addInPlace(result.positionGradients[1]);
            gradient[pos2Block].addInPlace(result.positionGradients[2]);
            
            // Add theta gradients
            gradient[theta0Block].addInPlace(result.thetaGradients[0]);
            gradient[theta1Block].addInPlace(result.thetaGradients[1]);
            gradient[theta2Block].addInPlace(result.thetaGradients[2]);
            
            // Add hessian blocks
            const posBlocks = [pos0Block, pos1Block, pos2Block];
            const thetaBlocks = [theta0Block, theta1Block, theta2Block];
            
            // Position-position hessian
            for (let j = 0; j < 3; j++) {
                for (let k = 0; k < 3; k++) {
                    this.bsm.addBlockAt(posBlocks[j], posBlocks[k], result.positionHessian[j][k]);
                }
            }
            
            // Theta-theta hessian
            for (let j = 0; j < 3; j++) {
                for (let k = 0; k < 3; k++) {
                    this.bsm.addBlockAt(thetaBlocks[j], thetaBlocks[k], result.thetaHessian[j][k]);
                }
            }
            
            // Cross hessian (position-theta coupling)
            for (let j = 0; j < 3; j++) {
                for (let k = 0; k < 3; k++) {
                    this.bsm.addBlockAt(posBlocks[j], thetaBlocks[k], result.crossHessian[j][k]);
                    this.bsm.addBlockAt(thetaBlocks[j], posBlocks[k], result.crossHessian[k][j]);
                }
            }
        }
        */
        
        // Step 5: Add mass matrix (only for position blocks)
        const massMatrix = Matrix3x3.identity().scale(this.geometry.vertexMass / (deltaTime * deltaTime));
        const thetaInertiaMatrix = Matrix3x3.identity().scale(this.geometry.vertexMass * this.geometry.radius * this.geometry.radius / (deltaTime * deltaTime));
        
        for (let i = 0; i < numVertices; i++) {
            const posBlock = this.geometry.getPositionBlockIndex(i);
            const thetaBlock = this.geometry.getThetaBlockIndex(i);
            
            this.bsm.addBlockAt(posBlock, posBlock, massMatrix);
            this.bsm.addBlockAt(thetaBlock, thetaBlock, thetaInertiaMatrix);
        }
        
        // Step 6: Add gravity forces (only for position blocks)
        for (let i = 0; i < numVertices; i++) {
            const posBlock = this.geometry.getPositionBlockIndex(i);
            gradient[posBlock].subtractInPlace(this.gravity.scale(this.geometry.vertexMass));
            totalEnergy -= this.geometry.vertexMass * Vector3.Dot(this.gravity, this.geometry.getPosition(i));
        }
        
        // Step 7: Handle fixed blocks
        for (const fixedBlock of this.geometry.fixedBlocks) {
            gradient[fixedBlock] = new Vector3(0, 0, 0);
            this.velocities[fixedBlock] = new Vector3(0, 0, 0);
            this.bsm.setFixed(fixedBlock);
        }
        
        // Step 8: Solve linear system for all blocks
        const delta = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);
        
        // Step 9: Update velocities and q
        for (let i = 0; i < numBlocks; i++) {
            if (!this.geometry.isBlockFixed(i)) {
                this.velocities[i].subtractInPlace(delta[i].scale(1 / deltaTime));
                this.geometry.q[i].subtractInPlace(delta[i]);
            }
        }
    }

    getPositions(): Vector3[] {
        return this.geometry.getPositions();
    }

    getThetas(): number[] {
        return this.geometry.getThetas();
    }

    getQ(): Vector3[] {
        return this.geometry.q.map(q => q.clone());
    }
}

// Utility function to create a simple rod
export function createRod(
    numVertices: number, 
    length: number = 5.0,
    stretchStiffness: number = 1000,
    bendStiffness: number = 100,
    twistStiffness: number = 100,
    vertexMass: number = 1.0,
    radius: number = 0.1
): DERGeometry {
    const positions = [];
    const thetas = [];
    
    // Create straight rod along x-axis
    for (let i = 0; i < numVertices; i++) {
        const x = (i / (numVertices - 1)) * length;
        positions.push(new Vector3(x, 5, 0));
        thetas.push(0); // Initial twist angles
    }
    
    const geometry = new DERGeometry(
        positions, 
        thetas,
        stretchStiffness,
        bendStiffness,
        twistStiffness,
        vertexMass,
        radius
    );
    
    // Fix one end of the rod
    geometry.setFixedVertex(0);
    
    return geometry;
}