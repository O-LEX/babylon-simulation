import { Vector3 } from "@babylonjs/core";
import { Matrix3x3 } from "./util";
import { Geometry } from "./geometry";
import { Params } from "./params";

export class MultiResVBDSolver {
    numVertices: number;
    pos: Float32Array; // numVertices * 3
    prevPos: Float32Array; // used to calculate velocity
    inertiaPos: Float32Array;
    vel: Float32Array;
    masses: Float32Array; // numVertices
    fixedVertices: Uint8Array; // numVertices, 1 if fixed, 0 if free

    numEdges: number;
    edges: Uint32Array; // two vertices per edge
    stiffnesses: Float32Array; // numEdges
    restLengths: Float32Array; // numEdges
    edgeTypes: Uint8Array; // numEdges, 0 for 1-hop (original), 1 for 2-hop (bending), 2 for 3-hop (long-range)

    vertexToEdgeStart: Uint32Array; // numVertices + 1
    vertexToEdgeIndices: Uint32Array; // numEdges * 2

    params: Params;

    constructor(geometry: Geometry, params: Params) {
        this.numVertices = geometry.pos.length / 3;
        this.pos = new Float32Array(geometry.pos);
        this.prevPos = new Float32Array(this.numVertices * 3);
        this.inertiaPos = new Float32Array(this.numVertices * 3);
        this.vel = new Float32Array(this.numVertices * 3);
        this.masses = new Float32Array(geometry.masses);
        this.fixedVertices = new Uint8Array(geometry.fixedVertices);
        this.params = params;

        // Step 1: Process original edges
        const originalEdges = Array.from(geometry.edges) as number[];
        const originalStiffnesses = Array.from(geometry.stiffnesses) as number[];
        
        // Step 2: Build adjacency list from original edges
        const adjacencyList = this.buildAdjacencyList(originalEdges);
        
        // Step 3: Find and add 2-hop edges
        const { newEdges: twoHopEdges, newStiffnesses: twoHopStiffnesses } = this.find2HopEdges(
            adjacencyList, 
            originalEdges, 
            originalStiffnesses
        );
        
        // Step 4: Find and add 3-hop edges
        const { newEdges: threeHopEdges, newStiffnesses: threeHopStiffnesses } = this.find3HopEdges(
            adjacencyList, 
            [...originalEdges, ...twoHopEdges], 
            [...originalStiffnesses, ...twoHopStiffnesses]
        );
        
        // Debug: Calculate edge counts
        const numOriginalEdges = originalEdges.length / 2;
        const numTwoHopEdges = twoHopEdges.length / 2;
        const numThreeHopEdges = threeHopEdges.length / 2;
        console.log("Edge Debug Info:");
        console.log(`  Original edges (1-hop): ${numOriginalEdges}`);
        console.log(`  New edges (2-hop): ${numTwoHopEdges}`);
        console.log(`  New edges (3-hop): ${numThreeHopEdges}`);
        console.log(`  Total edges: ${numOriginalEdges + numTwoHopEdges + numThreeHopEdges}`);
        
        // Step 5: Combine all edges
        this.edges = new Uint32Array([...originalEdges, ...twoHopEdges, ...threeHopEdges]);
        this.stiffnesses = new Float32Array([...originalStiffnesses, ...twoHopStiffnesses, ...threeHopStiffnesses]);
        this.numEdges = this.edges.length / 2;
        
        // Step 6: Initialize edge types (0 for 1-hop, 1 for 2-hop, 2 for 3-hop)
        this.edgeTypes = new Uint8Array(this.numEdges);
        
        // Mark original edges as 1-hop (0)
        for (let e = 0; e < numOriginalEdges; e++) {
            this.edgeTypes[e] = 0;
        }
        
        // Mark 2-hop edges as 2-hop (1)
        for (let e = 0; e < numTwoHopEdges; e++) {
            this.edgeTypes[numOriginalEdges + e] = 1;
        }
        
        // Mark 3-hop edges as 3-hop (2)
        for (let e = 0; e < numThreeHopEdges; e++) {
            this.edgeTypes[numOriginalEdges + numTwoHopEdges + e] = 2;
        }
        
        // Step 7: Calculate rest lengths for all edges
        this.restLengths = new Float32Array(this.numEdges);
        this.calculateRestLengths();
        
        // Step 7: Build final vertex-to-edge structure
        this.vertexToEdgeStart = new Uint32Array(this.numVertices + 1);
        this.vertexToEdgeIndices = new Uint32Array(this.numEdges * 2);
        this.buildVertexToEdgeStructure();
    }

    // Build adjacency list from edges array
    private buildAdjacencyList(edges: number[]): number[][] {
        const adjacencyList: number[][] = Array(this.numVertices).fill(null).map(() => []);
        
        for (let e = 0; e < edges.length / 2; e++) {
            const v0 = edges[e * 2];
            const v1 = edges[e * 2 + 1];
            
            adjacencyList[v0].push(v1);
            adjacencyList[v1].push(v0);
        }
        
        return adjacencyList;
    }

    // Find 2-hop edges and calculate their stiffnesses
    private find2HopEdges(
        adjacencyList: number[][], 
        originalEdges: number[], 
        originalStiffnesses: number[]
    ): { newEdges: number[], newStiffnesses: number[] } {
        const newEdges: number[] = [];
        const newStiffnesses: number[] = [];
        
        // Create a set to track existing edges to avoid duplicates
        const edgeSet = new Set<string>();
        
        // Add original edges to the set
        for (let e = 0; e < originalEdges.length / 2; e++) {
            const v0 = originalEdges[e * 2];
            const v1 = originalEdges[e * 2 + 1];
            const key = v0 < v1 ? `${v0},${v1}` : `${v1},${v0}`;
            edgeSet.add(key);
        }
        
        // Create stiffness lookup for original edges
        const stiffnessMap = new Map<string, number>();
        for (let e = 0; e < originalEdges.length / 2; e++) {
            const v0 = originalEdges[e * 2];
            const v1 = originalEdges[e * 2 + 1];
            const key = v0 < v1 ? `${v0},${v1}` : `${v1},${v0}`;
            stiffnessMap.set(key, originalStiffnesses[e]);
        }
        
        // Find 2-hop edges
        for (let vertex = 0; vertex < this.numVertices; vertex++) {
            const directNeighbors = adjacencyList[vertex];
            
            for (const neighbor of directNeighbors) {
                const secondHopNeighbors = adjacencyList[neighbor];
                
                for (const secondHop of secondHopNeighbors) {
                    // Check conditions: not self, not direct neighbor, edge doesn't exist
                    if (secondHop !== vertex && 
                        !directNeighbors.includes(secondHop)) {
                        
                        const key = vertex < secondHop ? `${vertex},${secondHop}` : `${secondHop},${vertex}`;
                        
                        if (!edgeSet.has(key)) {
                            // Add new edge
                            newEdges.push(vertex, secondHop);
                            edgeSet.add(key);
                            
                            // Calculate stiffness: 1/k = 1/k0 + 1/k1
                            const key0 = vertex < neighbor ? `${vertex},${neighbor}` : `${neighbor},${vertex}`;
                            const key1 = neighbor < secondHop ? `${neighbor},${secondHop}` : `${secondHop},${neighbor}`;
                            
                            const k0 = stiffnessMap.get(key0) || 1.0;
                            const k1 = stiffnessMap.get(key1) || 1.0;
                            const newStiffness = 1 / (1/k0 + 1/k1);
                            
                            newStiffnesses.push(newStiffness);
                        }
                    }
                }
            }
        }
        
        return { newEdges, newStiffnesses };
    }

    // Find 3-hop edges and calculate their stiffnesses
    private find3HopEdges(
        adjacencyList: number[][], 
        existingEdges: number[], 
        existingStiffnesses: number[]
    ): { newEdges: number[], newStiffnesses: number[] } {
        const newEdges: number[] = [];
        const newStiffnesses: number[] = [];
        
        // Create a set to track existing edges to avoid duplicates
        const edgeSet = new Set<string>();
        
        // Add existing edges to the set
        for (let e = 0; e < existingEdges.length / 2; e++) {
            const v0 = existingEdges[e * 2];
            const v1 = existingEdges[e * 2 + 1];
            const key = v0 < v1 ? `${v0},${v1}` : `${v1},${v0}`;
            edgeSet.add(key);
        }
        
        // Create stiffness lookup for existing edges
        const stiffnessMap = new Map<string, number>();
        for (let e = 0; e < existingEdges.length / 2; e++) {
            const v0 = existingEdges[e * 2];
            const v1 = existingEdges[e * 2 + 1];
            const key = v0 < v1 ? `${v0},${v1}` : `${v1},${v0}`;
            stiffnessMap.set(key, existingStiffnesses[e]);
        }
        
        // Find 3-hop edges
        for (let vertex = 0; vertex < this.numVertices; vertex++) {
            const directNeighbors = adjacencyList[vertex];
            
            for (const firstHop of directNeighbors) {
                const secondHopNeighbors = adjacencyList[firstHop];
                
                for (const secondHop of secondHopNeighbors) {
                    if (secondHop === vertex) continue; // Skip self
                    
                    const thirdHopNeighbors = adjacencyList[secondHop];
                    
                    for (const thirdHop of thirdHopNeighbors) {
                        // Check conditions: not self, not already connected
                        if (thirdHop !== vertex && 
                            !directNeighbors.includes(thirdHop)) {
                            
                            const key = vertex < thirdHop ? `${vertex},${thirdHop}` : `${thirdHop},${vertex}`;
                            
                            if (!edgeSet.has(key)) {
                                // Add new edge
                                newEdges.push(vertex, thirdHop);
                                edgeSet.add(key);
                                
                                // Calculate stiffness: 1/k = 1/k0 + 1/k1 + 1/k2
                                const key0 = vertex < firstHop ? `${vertex},${firstHop}` : `${firstHop},${vertex}`;
                                const key1 = firstHop < secondHop ? `${firstHop},${secondHop}` : `${secondHop},${firstHop}`;
                                const key2 = secondHop < thirdHop ? `${secondHop},${thirdHop}` : `${thirdHop},${secondHop}`;
                                
                                const k0 = stiffnessMap.get(key0) || 1.0;
                                const k1 = stiffnessMap.get(key1) || 1.0;
                                const k2 = stiffnessMap.get(key2) || 1.0;
                                const newStiffness = 1 / (1/k0 + 1/k1 + 1/k2);
                                
                                newStiffnesses.push(newStiffness);
                            }
                        }
                    }
                }
            }
        }
        
        return { newEdges, newStiffnesses };
    }

    // Calculate rest lengths for all edges
    private calculateRestLengths(): void {
        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];

            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);

            this.restLengths[e] = Vector3.Distance(p0, p1);
        }
    }

    // Build vertex-to-edge adjacency structure (CSR format)
    private buildVertexToEdgeStructure(): void {
        // Count edges per vertex
        const edgeCountPerVertex = new Uint32Array(this.numVertices);
        for (let e = 0; e < this.numEdges; e++) {
            edgeCountPerVertex[this.edges[e * 2]]++;
            edgeCountPerVertex[this.edges[e * 2 + 1]]++;
        }

        // Build start indices
        this.vertexToEdgeStart[0] = 0;
        for (let i = 0; i < this.numVertices; i++) {
            this.vertexToEdgeStart[i + 1] = this.vertexToEdgeStart[i] + edgeCountPerVertex[i];
        }

        // Fill edge indices
        const currentOffset = new Uint32Array(this.numVertices);
        for (let i = 0; i < this.numVertices; i++) {
            currentOffset[i] = this.vertexToEdgeStart[i];
        }

        for (let e = 0; e < this.numEdges; e++) {
            const v0 = this.edges[e * 2];
            const v1 = this.edges[e * 2 + 1];

            this.vertexToEdgeIndices[currentOffset[v0]++] = e;
            this.vertexToEdgeIndices[currentOffset[v1]++] = e;
        }
    }

    step() {
        const dt = this.params.dt / this.params.numSubsteps;
        const g = this.params.g;

        for (let step = 0; step < this.params.numSubsteps; step++) {
            this.forward(dt, g);
            for (let itr = 0; itr < this.params.numIterations; itr++) {
                this.solve(dt, itr, this.params.numIterations);
            }
            this.updateVel(dt);
        }
    }

    forward(dt: number, g: Vector3) {
        this.prevPos.set(this.pos);

        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) this.setVector3(this.inertiaPos, i, this.getVector3(this.pos, i));
            else {
                const p = this.getVector3(this.pos, i);
                const v = this.getVector3(this.vel, i);
                
                v.addInPlace(g.scale(dt)); // gravity is included in inertia
                p.addInPlace(v.scale(dt));
                this.setVector3(this.pos, i, p);
                this.setVector3(this.inertiaPos, i, p);
            }
        }
    }

    solve(dt: number, itr: number, numIterations: number) {
        const invDt2 = 1 / (dt * dt);

        // Calculate edge type weights using exponential function
        // t ranges from 0.0 to 1.0 over iterations
        const t = numIterations > 1 ? itr / (numIterations - 1) : 1.0;
        
        // Exponential transition parameters
        const alpha = 1.0; // Controls steepness of transition (higher = steeper)
        
        // Multi-hop weight calculation
        // Early iterations: focus on long-range constraints (3-hop, 2-hop)
        // Later iterations: focus on structural constraints (1-hop)
        const oneHopWeight = (Math.exp(alpha * t) - 1) / (Math.exp(alpha) - 1);
        const twoHopWeight = (Math.exp(alpha * (1 - t)) - 1) / (Math.exp(alpha) - 1);
        const threeHopWeight = Math.exp(-alpha * t); // Decays exponentially

        let totalEnergy = 0;

        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;

            const mass = this.masses[i];
            const p = this.getVector3(this.pos, i);
            const inertiaP = this.getVector3(this.inertiaPos, i);

            totalEnergy += 0.5 * mass * Vector3.DistanceSquared(p, inertiaP) * invDt2; // inertia energy
            let gradient = p.subtract(inertiaP).scale(mass * invDt2); // intertia term

            // gradient.subtractInPlace(g.scale(mass)); // gravity force if you don't want to include it in inertia
            let hessian = Matrix3x3.identity().scale(mass * invDt2); // mass matrix

            for (let j = this.vertexToEdgeStart[i]; j < this.vertexToEdgeStart[i + 1]; j++) {
                const e = this.vertexToEdgeIndices[j];
                const id0 = this.edges[e * 2];
                const id1 = this.edges[e * 2 + 1];
                const stiffness = this.stiffnesses[e];
                const edgeType = this.edgeTypes[e]; // 0 for 1-hop, 1 for 2-hop, 2 for 3-hop
                
                // Apply weight based on edge type and iteration
                let edgeWeight: number;
                if (edgeType === 0) {
                    edgeWeight = oneHopWeight;    // 1-hop edges
                } else if (edgeType === 1) {
                    edgeWeight = twoHopWeight;    // 2-hop edges
                } else {
                    edgeWeight = threeHopWeight;  // 3-hop edges
                }
                
                // Skip if weight is zero
                if (edgeWeight < 1e-8) continue;
                
                const p0 = this.getVector3(this.pos, id0);
                const p1 = this.getVector3(this.pos, id1);
                const restLength = this.restLengths[e];

                const diff = p1.subtract(p0);
                const length = diff.length();
                if (length < 1e-8) continue; // Avoid division by zero

                const C = length - restLength;
                if (Math.abs(C) < 1e-8) continue; // Skip if no change

                const weightedStiffness = stiffness * edgeWeight;
                totalEnergy += 0.5 * weightedStiffness * C * C; // Weighted potential energy
                const u01 = diff.normalize();

                if (id0 === i) gradient.subtractInPlace(u01.scale(weightedStiffness * C));
                else gradient.addInPlace(u01.scale(weightedStiffness * C));

                const uu = Matrix3x3.outerProduct(u01, u01);
                const du = Matrix3x3.identity().subtract(uu).scale(1 / length);
                const o = uu.add(du.scale(C)).scale(weightedStiffness);

                hessian = hessian.add(o);
            }

            const delta = hessian.solve(gradient);

            this.setVector3(this.pos, i, p.subtract(delta));
        }
        // console.log("Total Energy:", totalEnergy);
    }

    updateVel(dt: number) {
        const invDt = 1 / dt;
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
