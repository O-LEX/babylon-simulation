import { Vector3 } from "@babylonjs/core";
import { Geometry } from "./geometry";
import { Params } from "./params";

export class XPBDSolver {
    numVertices: number;
    pos: Float32Array;
    prevPos: Float32Array;
    vel: Float32Array;
    invMass: Float32Array;

    numEdges: number; // Constraints
    edges: Uint32Array; // should be constraints but using edges for compatibility
    compliances: Float32Array; // inverse stiffnesses
    restLengths: Float32Array; // rest lengths of edges

    params: Params;

    constructor(geometry: Geometry, params: Params) {
        this.numVertices = geometry.pos.length / 3;
        this.pos = new Float32Array(geometry.pos);
        this.prevPos = new Float32Array(this.numVertices * 3);
        this.vel = new Float32Array(this.numVertices * 3);
        this.invMass = new Float32Array(this.numVertices);
        this.numEdges = geometry.edges.length / 2;
        this.edges = new Uint32Array(geometry.edges);
        this.compliances = new Float32Array(this.numEdges);
        this.restLengths = new Float32Array(this.numEdges);
        this.params = params;

        for (let e = 0; e < this.numEdges; e++) {
            const id0 = this.edges[e * 2];
            const id1 = this.edges[e * 2 + 1];

            const p0 = this.getVector3(this.pos, id0);
            const p1 = this.getVector3(this.pos, id1);

            this.restLengths[e] = Vector3.Distance(p0, p1);
            this.compliances[e] = 1.0 / geometry.stiffnesses[e];
        }

        for (let i = 0; i < this.numVertices; i++) {
            if (geometry.fixedVertices[i] === 1) {
                this.invMass[i] = 0; // Fixed vertices have infinite mass
            } else {
                this.invMass[i] = 1.0 / geometry.masses[i];
            }
        }
    }

    step() {
        for (let step = 0; step < this.params.numSubsteps; step++) {
            this.solve();
        }
    }

    solve() {
        this.prevPos.set(this.pos);

        const g = this.params.g;
        const dt = this.params.dt / this.params.numSubsteps;
        const invDt = 1.0 / dt;
        const invDt2 = 1.0 / (dt * dt);

        for (let i = 0; i < this.numVertices; i++) {
            if (this.invMass[i] === 0) continue;
            let p = this.getVector3(this.prevPos, i);
            let v = this.getVector3(this.vel, i);
            v.addInPlace(g.scale(dt)); // Apply gravity
            p.addInPlace(v.scale(dt)); // Update position with velocity
            this.setVector3(this.pos, i, p);
        }

        for (let itr = 0; itr < this.params.numIterations; itr++) {
            for (let e = 0; e < this.numEdges; e++) {
                const id0 = this.edges[e * 2];
                const id1 = this.edges[e * 2 + 1];
                const w0 = this.invMass[id0];
                const w1 = this.invMass[id1];
                if (w0 + w1 === 0) continue; // Both vertices are fixed
                const p0 = this.getVector3(this.pos, id0);
                const p1 = this.getVector3(this.pos, id1);
                const diff = p1.subtract(p0);
                const length = diff.length();
                if (length < 1e-6) continue; // Avoid division by zero
                const restLength = this.restLengths[e];
                const compliance = this.compliances[e];

                const u01 = diff.normalize();
                const C = length - restLength;
                const alpha = compliance * invDt2;
                const s = -C / (w0 + w1 + alpha);
                const s0 = s * w0;
                const s1 = s * w1;

                this.setVector3(this.pos, id0, p0.subtract(u01.scale(s0)));
                this.setVector3(this.pos, id1, p1.add(u01.scale(s1)));
            }
        }

        for (let i = 0; i < this.numVertices; i++) {
            if (this.invMass[i] === 0) continue; // Skip fixed vertices
            const p = this.getVector3(this.pos, i);
            const v = p.subtract(this.getVector3(this.prevPos, i)).scale(invDt);
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