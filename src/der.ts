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
    private numVertices: number = 0;
    
    // Working arrays for CG solver
    private p: Vector3[] = [];
    private Ap: Vector3[] = [];

    initialize(row2col: number[], idx2col: number[]): void {
        this.row2idx = [...row2col];
        this.idx2col = [...idx2col];
        this.idx2val = new Array(idx2col.length).fill(null).map(() => new Matrix3x3());
        this.numVertices = row2col.length - 1;
        
        // Initialize working arrays
        this.p = new Array(this.numVertices).fill(null).map(() => new Vector3(0, 0, 0));
        this.Ap = new Array(this.numVertices).fill(null).map(() => new Vector3(0, 0, 0));
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

    setFixed(i_vtx: number): void {
        for (let j_vtx = 0; j_vtx < this.row2idx.length - 1; j_vtx++) {
            if (j_vtx === i_vtx) {
                // For the fixed vertex row
                for (let idx = this.row2idx[j_vtx]; idx < this.row2idx[j_vtx + 1]; idx++) {
                    if (this.idx2col[idx] === i_vtx) {
                        this.idx2val[idx] = this.idx2val[idx].add(Matrix3x3.identity());
                    } else {
                        this.idx2val[idx] = new Matrix3x3(); // Zero
                    }
                }
            } else {
                // For other rows, zero out the column corresponding to fixed vertex
                for (let idx = this.row2idx[j_vtx]; idx < this.row2idx[j_vtx + 1]; idx++) {
                    if (this.idx2col[idx] === i_vtx) {
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


// Rod element representing the elastic rod structure
class RodElement {
    vertices: number[]; // Vertex indices that make up this element
    restLength: number; // Rest length of the element
    restTwist: number; // Rest twist angle
    restCurvature: Vector3; // Rest curvature vector

    constructor(vertices: number[], restLength: number, restTwist: number = 0, restCurvature?: Vector3) {
        this.vertices = [...vertices];
        this.restLength = restLength;
        this.restTwist = restTwist;
        this.restCurvature = restCurvature || new Vector3(0, 0, 0);
    }
}

// Discrete Elastic Rod geometry
export class DERGeometry {
    positions: Vector3[];
    initialPositions: Vector3[];
    thetas: number[]; // Twist angles at each vertex (discrete viscous thread model)
    elements: RodElement[];
    fixedVertices: Set<number>;
    
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
        this.positions = positions.map(p => p.clone());
        this.initialPositions = positions.map(p => p.clone());
        this.thetas = [...thetas];
        this.fixedVertices = new Set();
        this.stretchStiffness = stretchStiffness;
        this.bendStiffness = bendStiffness;
        this.twistStiffness = twistStiffness;
        this.vertexMass = vertexMass;
        this.radius = radius;
        
        this.elements = this.createElements();
    }

    private createElements(): RodElement[] {
        const elements: RodElement[] = [];
        
        for (let i = 0; i < this.positions.length - 1; i++) {
            const restLength = Vector3.Distance(this.initialPositions[i], this.initialPositions[i + 1]);
            elements.push(new RodElement([i, i + 1], restLength));
        }
        
        return elements;
    }

    setFixedVertex(vertexIndex: number): void {
        this.fixedVertices.add(vertexIndex);
    }

    setFixedVertices(vertexIndices: number[]): void {
        for (const index of vertexIndices) {
            this.fixedVertices.add(index);
        }
    }

    isFixed(vertexIndex: number): boolean {
        return this.fixedVertices.has(vertexIndex);
    }

    getNumVertices(): number {
        return this.positions.length;
    }

    getNumElements(): number {
        return this.elements.length;
    }
}

// Discrete Elastic Rod solver using discrete viscous thread model
export class DERSolver {
    private geometry: DERGeometry;
    private bsm: BlockSparseMatrix;
    private velocities: Vector3[];
    private thetaVelocities: number[];
    private gravity: Vector3 = new Vector3(0, -9.81, 0);

    constructor(geometry: DERGeometry) {
        this.geometry = geometry;
        this.bsm = new BlockSparseMatrix();
        
        // Create sparse matrix structure
        const structure = this.createSparseMatrixStructure(geometry);
        this.bsm.initialize(structure.row2idx, structure.idx2col);
        
        // Initialize velocities to zero
        this.velocities = new Array(geometry.getNumVertices())
            .fill(null).map(() => new Vector3(0, 0, 0));
        this.thetaVelocities = new Array(geometry.getNumVertices()).fill(0);
    }

    private createSparseMatrixStructure(geometry: DERGeometry): { row2idx: number[], idx2col: number[] } {
        const numVertices = geometry.getNumVertices();
        const adjacency = new Map<number, Set<number>>();
        
        // Initialize adjacency lists
        for (let i = 0; i < numVertices; i++) {
            adjacency.set(i, new Set([i])); // Self-connection
        }
        
        // Add element connections (more neighbors for bending and twisting)
        for (let i = 0; i < geometry.getNumElements(); i++) {
            const element = geometry.elements[i];
            for (let j = 0; j < element.vertices.length; j++) {
                for (let k = 0; k < element.vertices.length; k++) {
                    if (j !== k) {
                        adjacency.get(element.vertices[j])!.add(element.vertices[k]);
                    }
                }
            }
        }
        
        // Add extended neighbors for bending (i-1, i, i+1, i+2 pattern)
        for (let i = 0; i < numVertices - 2; i++) {
            for (let j = i; j <= i + 2; j++) {
                for (let k = i; k <= i + 2; k++) {
                    if (j < numVertices && k < numVertices) {
                        adjacency.get(j)!.add(k);
                    }
                }
            }
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

    // Bending energy computation using discrete viscous thread model
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
        
        const t1 = e1.normalize();
        const t2 = e2.normalize();
        
        // Curvature vector (discrete difference of tangents)
        const kappa = t2.subtract(t1).scale(2.0 / (l1 + l2));
        const kappaSquared = Vector3.Dot(kappa, kappa);
        
        const energy = 0.5 * stiffness * kappaSquared;
        
        // Gradients (simplified computation)
        const factor = stiffness * 2.0 / (l1 + l2);
        const grad0 = kappa.scale(-factor / l1);
        const grad1 = kappa.scale(factor * (1.0 / l1 + 1.0 / l2));
        const grad2 = kappa.scale(-factor / l2);
        
        const gradients = [grad0, grad1, grad2];
        
        // Simplified hessian (identity scaled by stiffness for stability)
        const H = Matrix3x3.identity().scale(stiffness * 0.1);
        const hessian = [
            [H, H.scale(-0.5), new Matrix3x3()],
            [H.scale(-0.5), H.scale(2), H.scale(-0.5)],
            [new Matrix3x3(), H.scale(-0.5), H]
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
    ): { energy: number, gradients: Vector3[], thetaGradients: number[] } {
        const e1 = p1.subtract(p0);
        const e2 = p2.subtract(p1);
        
        const l1 = e1.length();
        const l2 = e2.length();
        
        if (l1 < 1e-8 || l2 < 1e-8) {
            return {
                energy: 0,
                gradients: [new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Vector3(0, 0, 0)],
                thetaGradients: [0, 0, 0]
            };
        }
        
        // Discrete twist computation
        const avgLength = (l1 + l2) * 0.5;
        const twist = (theta2 - theta1) / avgLength - (theta1 - theta0) / avgLength;
        
        const energy = 0.5 * stiffness * twist * twist;
        
        // Gradients (twist affects both positions and theta values)
        const factor = stiffness * twist / avgLength;
        
        // Position gradients (simplified - twist primarily affects theta)
        const gradients = [
            new Vector3(0, 0, 0), // Simplified - twist primarily affects theta
            new Vector3(0, 0, 0),
            new Vector3(0, 0, 0)
        ];
        
        // Theta gradients
        const thetaGradients = [
            factor * (-1.0 / avgLength),
            factor * (2.0 / avgLength),
            factor * (-1.0 / avgLength)
        ];
        
        return { energy, gradients, thetaGradients };
    }

    step(deltaTime: number): void {
        const numVertices = this.geometry.getNumVertices();
        const gradient = new Array(numVertices).fill(null).map(() => new Vector3(0, 0, 0));
        const thetaGradient = new Array(numVertices).fill(0);
        
        this.bsm.setZero();
        
        // Step 1: Update positions using current velocities
        for (let i = 0; i < numVertices; i++) {
            if (!this.geometry.isFixed(i)) {
                this.geometry.positions[i].addInPlace(this.velocities[i].scale(deltaTime));
                this.geometry.thetas[i] += this.thetaVelocities[i] * deltaTime;
            }
        }
        
        let totalEnergy = 0;
        
        // Step 2: Process stretching forces
        for (let i = 0; i < this.geometry.getNumElements(); i++) {
            const element = this.geometry.elements[i];
            const v0 = element.vertices[0];
            const v1 = element.vertices[1];
            
            const result = this.computeStretchingForces(
                this.geometry.positions[v0],
                this.geometry.positions[v1],
                element.restLength,
                this.geometry.stretchStiffness
            );
            
            totalEnergy += result.energy;
            
            gradient[v0].addInPlace(result.gradients[0]);
            gradient[v1].addInPlace(result.gradients[1]);
            
            this.bsm.addBlockAt(v0, v0, result.hessian[0][0]);
            this.bsm.addBlockAt(v0, v1, result.hessian[0][1]);
            this.bsm.addBlockAt(v1, v0, result.hessian[1][0]);
            this.bsm.addBlockAt(v1, v1, result.hessian[1][1]);
        }
        
        // Step 3: Process bending forces
        for (let i = 0; i < numVertices - 2; i++) {
            const result = this.computeBendingForces(
                this.geometry.positions[i],
                this.geometry.positions[i + 1],
                this.geometry.positions[i + 2],
                this.geometry.bendStiffness
            );
            
            totalEnergy += result.energy;
            
            gradient[i].addInPlace(result.gradients[0]);
            gradient[i + 1].addInPlace(result.gradients[1]);
            gradient[i + 2].addInPlace(result.gradients[2]);
            
            // Add hessian blocks
            for (let j = 0; j < 3; j++) {
                for (let k = 0; k < 3; k++) {
                    this.bsm.addBlockAt(i + j, i + k, result.hessian[j][k]);
                }
            }
        }
        
        // Step 4: Process twisting forces
        for (let i = 0; i < numVertices - 2; i++) {
            const result = this.computeTwistingForces(
                this.geometry.positions[i],
                this.geometry.positions[i + 1],
                this.geometry.positions[i + 2],
                this.geometry.thetas[i],
                this.geometry.thetas[i + 1],
                this.geometry.thetas[i + 2],
                this.geometry.twistStiffness
            );
            
            totalEnergy += result.energy;
            
            gradient[i].addInPlace(result.gradients[0]);
            gradient[i + 1].addInPlace(result.gradients[1]);
            gradient[i + 2].addInPlace(result.gradients[2]);
            
            thetaGradient[i] += result.thetaGradients[0];
            thetaGradient[i + 1] += result.thetaGradients[1];
            thetaGradient[i + 2] += result.thetaGradients[2];
        }
        
        // Step 5: Add mass matrix
        const massMatrix = Matrix3x3.identity().scale(this.geometry.vertexMass / (deltaTime * deltaTime));
        for (let i = 0; i < numVertices; i++) {
            this.bsm.addBlockAt(i, i, massMatrix);
        }
        
        // Step 6: Add gravity forces
        for (let i = 0; i < numVertices; i++) {
            gradient[i].subtractInPlace(this.gravity.scale(this.geometry.vertexMass));
            totalEnergy -= this.geometry.vertexMass * Vector3.Dot(this.gravity, this.geometry.positions[i]);
        }
        
        // Step 7: Handle fixed vertices
        for (const fixedIndex of this.geometry.fixedVertices) {
            gradient[fixedIndex] = new Vector3(0, 0, 0);
            thetaGradient[fixedIndex] = 0;
            this.velocities[fixedIndex] = new Vector3(0, 0, 0);
            this.thetaVelocities[fixedIndex] = 0;
            this.bsm.setFixed(fixedIndex);
        }
        
        // Step 8: Solve linear system for positions
        const delta = this.bsm.conjugateGradientSolver(gradient, 100, 1e-6);
        
        // Step 9: Update velocities and positions
        for (let i = 0; i < numVertices; i++) {
            if (!this.geometry.isFixed(i)) {
                this.velocities[i].subtractInPlace(delta[i].scale(1 / deltaTime));
                this.geometry.positions[i].subtractInPlace(delta[i]);
                
                // Update theta velocities (simplified integration)
                this.thetaVelocities[i] -= thetaGradient[i] * deltaTime / this.geometry.vertexMass;
                this.geometry.thetas[i] -= thetaGradient[i] * deltaTime * deltaTime / this.geometry.vertexMass;
            }
        }
    }

    getPositions(): Vector3[] {
        return this.geometry.positions.map(p => p.clone());
    }

    getThetas(): number[] {
        return [...this.geometry.thetas];
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