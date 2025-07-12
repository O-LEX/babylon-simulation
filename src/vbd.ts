import { Vector3 } from "@babylonjs/core";
import { Matrix3x3 } from "./util";
import { Geometry } from "./geometry";
import { Params } from "./params";

export class VBDSolver {
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
        this.numEdges = geometry.edges.length / 2;
        this.edges = new Uint32Array(geometry.edges);
        this.stiffnesses = new Float32Array(geometry.stiffnesses);
        this.restLengths = new Float32Array(this.numEdges);
        this.vertexToEdgeStart = new Uint32Array(this.numVertices + 1);
        this.vertexToEdgeIndices = new Uint32Array(this.numEdges * 2);
        this.params = params;

        // Initialize restLength
        for (let e = 0; e < this.numEdges; e++) {

            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];

            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);

            this.restLengths[e] = Vector3.Distance(p0, p1);
        }

        // Build vertexToEdge adjacency arrays (CSR-like structure)
        const edgeCountPerVertex = new Uint32Array(this.numVertices);
        for (let e = 0; e < this.numEdges; e++) {
            edgeCountPerVertex[this.edges[e * 2]]++;
            edgeCountPerVertex[this.edges[e * 2 + 1]]++;
        }

        this.vertexToEdgeStart[0] = 0;
        for (let i = 0; i < this.numVertices; i++) {
            this.vertexToEdgeStart[i + 1] = this.vertexToEdgeStart[i] + edgeCountPerVertex[i];
        }

        // Temporary offset counters for filling vertexToEdgeIndices
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
        this.forward();
        this.solve();
        this.updateVel();
    }

    forward() {
        this.prevPos.set(this.pos);

        const dt = this.params.dt;
        const g = this.params.g;

        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue; // Skip fixed vertices

            let p = this.getVector3(this.pos, i);
            let v = this.getVector3(this.vel, i);

            p.addInPlace(v.scale(dt));
            this.setVector3(this.pos, i, p);
            this.setVector3(this.inertiaPos, i, p);
        }
    }

    solve() {
        const dt = this.params.dt;
        const invDt2 = 1 / (dt * dt);

        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;

            const mass = this.masses[i];
            const p = this.getVector3(this.pos, i);
            const inertiaP = this.getVector3(this.inertiaPos, i);

            let f = inertiaP.subtract(p).scale(mass * invDt2);
            let h = Matrix3x3.identity().scale(mass * invDt2);

            for (let j = this.vertexToEdgeStart[i]; j < this.vertexToEdgeStart[i + 1]; j++) {
                const e = this.vertexToEdgeIndices[j];
                const id0 = this.edges[e * 2];
                const id1 = this.edges[e * 2 + 1];
                
                const p0 = this.getVector3(this.pos, id0);
                const p1 = this.getVector3(this.pos, id1);
                const restLength = this.restLengths[e];

                const edgeDir = p1.subtract(p0);
                const length = edgeDir.length();
                if (length < 1e-8) continue; // Avoid division by zero

                const deltaLength = length - restLength;
                if (Math.abs(deltaLength) < 1e-8) continue; // Skip if no change

                const stiffness = this.stiffnesses[e];
                const force = edgeDir.scale(stiffness * deltaLength / length);

                if (id0 === i) f.addInPlace(force);
                else f.subtractInPlace(force);

                const I = Matrix3x3.identity();
                const outer = Matrix3x3.outerProduct(edgeDir, edgeDir);
                const hessian = I.scale(stiffness * (1- restLength / length)).add(outer.scale(stiffness * restLength / (length * length * length)));

                h = h.add(hessian);
            }

            const dx = h.solve(f);

            this.setVector3(this.pos, i, p.subtract(dx));
        }
    }

    updateVel() {
        const dt = this.params.dt;
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
