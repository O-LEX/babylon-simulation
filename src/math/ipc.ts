import { Optimizable } from "./lbfgs";
import * as PT from "./PointTriangleDistance";
import { Vector3 } from "@babylonjs/core";

export class IPCOptimizable implements Optimizable {
    private r: number;
    private y: number[];

    constructor(r: number, y: number[]) {
        this.r = r;
        this.y = y;
    }

    diffsquared(parameters: number[]): number {
        let diff = 0;
        for (let i = 0; i < this.y.length; i++) {
            diff += (this.y[i] - parameters[i]) * (this.y[i] - parameters[i]);
        }
        return diff;
    }

    getValue(parameters: number[]): number {
        const z0 = new Vector3(parameters[0], parameters[1], parameters[2]);
        const z1 = new Vector3(parameters[3], parameters[4], parameters[5]);
        const z2 = new Vector3(parameters[6], parameters[7], parameters[8]);
        const d2 = PT.val(z0, z1, z2);
        const r2 = this.r * this.r;
        return this.r/8*(d2/r2 - 1)*Math.log(d2/r2) + 1/2*this.diffsquared(parameters);
    }

    getGradient(parameters: number[], gradient: number[]): number[] {
        const z0 = new Vector3(parameters[0], parameters[1], parameters[2]);
        const z1 = new Vector3(parameters[3], parameters[4], parameters[5]);
        const z2 = new Vector3(parameters[6], parameters[7], parameters[8]);
        const d2 = PT.val(z0, z1, z2);
        const r2 = this.r * this.r;
        const grad = PT.grad(z0, z1, z2);
        const temp = 1/8/this.r * (Math.log(d2/r2) + d2/r2 - 1);
        for (let i = 0; i < gradient.length; i++) {
            gradient[i] = temp * grad[i] + (this.y[i] - parameters[i]);
        }
        return gradient;
    }
}