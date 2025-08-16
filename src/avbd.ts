import { Geometry } from "./geometry";
import { Params } from "./params";
import { VBDSolver } from "./vbd";
import { Vector3 } from "@babylonjs/core";
import { Matrix3x3 } from "./math/util";

export class AVBDSolver extends VBDSolver {
    lambdas: Float32Array; 
    adaptiveStiffnesses: Float32Array; // numEdges
    alpha: number = 0.95; // stiffness adaptation factor
    beta: number = 10; // stiffness adaptation factor
    gamma: number = 0.99; // stiffness warm start factor

    constructor(geometry: Geometry, params: Params) {
        super(geometry, params);
        this.adaptiveStiffnesses = new Float32Array(this.numEdges).fill(1.0);
        this.lambdas = new Float32Array(this.numEdges);
    }

    override step() {
        this.warmStart();
        super.step();
    }

    warmStart() {
        for (let e = 0; e < this.numEdges; e++) {
            this.lambdas[e] = this.alpha * this.gamma * this.lambdas[e];
            this.adaptiveStiffnesses[e] = this.gamma * this.adaptiveStiffnesses[e]; // I have no idea about k_start
        }
    }

    override solve(dt: number) {
        const invDt2 = 1 / (dt * dt);

        // skip colorization

        // primal update
        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;

            const mass = this.masses[i];
            const p = this.getVector3(this.pos, i);
            const inertiaP = this.getVector3(this.inertiaPos, i);

            let f = p.subtract(inertiaP).scale(-mass * invDt2); // intertia term
            let H = Matrix3x3.identity().scale(mass * invDt2); // mass matrix

            for (let j = this.vertexToEdgeStart[i]; j < this.vertexToEdgeStart[i + 1]; j++) {
                const e = this.vertexToEdgeIndices[j];
                const id0 = this.edges[e * 2];
                const id1 = this.edges[e * 2 + 1];
                const p0 = this.getVector3(this.pos, id0);
                const p1 = this.getVector3(this.pos, id1);
                const diff = p1.subtract(p0);
                const length = diff.length();
                if (length < 1e-8) continue; // Avoid division by zero
                const u01 = diff.scale(1 / length);

                const stiffness = this.adaptiveStiffnesses[e];
                const restLength = this.restLengths[e];
                let lambda = this.lambdas[e];
                if (stiffness !== Infinity) lambda = 0;

                const C = length - restLength;

                const force = u01.scale(stiffness * C + lambda);

                if (id0 === i) f.addInPlace(force);
                else f.subtractInPlace(force);

                const uu = Matrix3x3.outerProduct(u01, u01);
                const du = Matrix3x3.identity().subtract(uu).scale(1 / length);
                const hess = uu.add(du.scale(C)).scale(stiffness).add(du.scale(lambda));

                H = H.add(hess);
            }

            const delta = H.solve(f);
            this.setVector3(this.pos, i, p.add(delta));
        }

        // dual update
        for (let e = 0; e < this.numEdges; e++) {
            if (this.stiffnesses[e] === Infinity) {
                this.lambdas[e] = this.alpha * this.gamma * this.lambdas[e];
                this.adaptiveStiffnesses[e] += this.gamma * this.adaptiveStiffnesses[e];
            } else {
                this.adaptiveStiffnesses[e] = Math.min(this.stiffnesses[e], this.adaptiveStiffnesses[e] * this.beta);
            }
        }
    }
}
