import { Geometry } from "./geometry";
import { Params } from "./params";
import { VBDSolver } from "./vbd";
import { Vector3 } from "@babylonjs/core";
import { Matrix3x3 } from "./util";

export class AVBDSolver extends VBDSolver {
    adaptiveStiffness: Float32Array; // numEdges
    lambdas: Float32Array; 
    alpha: number = 0.95; // stiffness adaptation factor
    beta: number = 10; // stiffness adaptation factor
    gamma: number = 0.99; // stiffness warm start factor

    constructor(geometry: Geometry, params: Params) {
        super(geometry, params);
        this.adaptiveStiffness = new Float32Array(this.numEdges).fill(1.0);
        this.lambdas = new Float32Array(this.numEdges);
    }

    override step() {
        this.warmStart();
        super.step();
    }

    warmStart() {
        for (let e = 0; e < this.numEdges; e++) {
            this.adaptiveStiffness[e] = Math.max(this.adaptiveStiffness[e] * this.gamma, 1.0);
            this.lambdas[e] = this.alpha * this.gamma * this.lambdas[e];
        }
    }

    override solve(dt: number) {
        const invDt2 = 1 / (dt * dt);

        for (let i = 0; i < this.numVertices; i++) {
            if (this.fixedVertices[i]) continue;

            const mass = this.masses[i];
            const p = this.getVector3(this.pos, i);
            const inertiaP = this.getVector3(this.inertiaPos, i);

            let gradient = p.subtract(inertiaP).scale(mass * invDt2); // intertia term

            let hessian = Matrix3x3.identity().scale(mass * invDt2); // mass matrix

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

                const stiffness = this.adaptiveStiffness[e];
                const restLength = this.restLengths[e];
                const lambda = this.lambdas[e];
                
                const C = length - restLength;

                const grad = u01.scale(stiffness * C + lambda);

                if (id0 === i) gradient.subtractInPlace(grad);
                else gradient.addInPlace(grad);

                const uu = Matrix3x3.outerProduct(u01, u01);
                const du = Matrix3x3.identity().subtract(uu).scale(1 / length);
                const hess = uu.add(du.scale(C)).scale(stiffness).add(du.scale(lambda));

                hessian = hessian.add(hess);

                // update lambda and stiffness
                this.lambdas[e] = stiffness * C + lambda;
                this.adaptiveStiffness[e] = Math.min(this.stiffnesses[e], stiffness + this.beta * Math.abs(C));
            }

            const delta = hessian.solve(gradient);
            this.setVector3(this.pos, i, p.subtract(delta));
        }
    }
}
