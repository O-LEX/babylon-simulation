import { Vector2, Vector3 } from "@babylonjs/core";

class Spring {
    pos: Vector3[];
    vel: Vector3[];
    mass: number[];
    isFixed: boolean[];
    edges: Vector2[];
    lengths: number[];
    stiffness: number;

    constructor(pos: Vector3[], mass: number[], stiffness: number) {
        this.pos = pos;
        this.vel = pos.map(() => new Vector3(0, 0, 0));
        this.mass = mass;
        this.isFixed = pos.map(() => false); // All points are initially not fixed
        this.isFixed[0] = true; // Fix the first point
        this.edges = [];
        this.lengths = [];
        this.stiffness = stiffness;

        for (let i = 0; i < pos.length - 1; i++) {
            this.edges.push(new Vector2(i, i + 1));
            this.lengths.push(Vector3.Distance(pos[i], pos[i + 1]));
        }
    }
}

export class SpringSim {
    spring: Spring;
    gravity: Vector3 = new Vector3(0, -9.8, 0);

    get springPos(): Vector3[] {
        return this.spring.pos;
    }

    constructor(pos: Vector3[], mass: number[], stiffness: number) {
        this.spring = new Spring(pos, mass, stiffness);
    }

    update(deltaTime: number) {
        const dt = deltaTime;
        const g = this.gravity;
        const pos = this.spring.pos;
        const vel = this.spring.vel;
        const mass = this.spring.mass;
        const isFixed = this.spring.isFixed;
        const edges = this.spring.edges;
        const stiffness = this.spring.stiffness;

        for (let i = 0; i < edges.length; i++) {
            const edge = edges[i];
            const p1 = pos[edge.x];
            const p2 = pos[edge.y];
            const length = this.spring.lengths[i];

            const dir = p2.subtract(p1);
            const currentLength = dir.length();
            if (currentLength === 0) continue;

            const forceMagnitude = stiffness * (currentLength - length);
            const forceDir = dir.normalize();
            const force = forceDir.scale(forceMagnitude);

            if (!isFixed[edge.x]) {
                vel[edge.x].addInPlace(force.scale(dt / mass[edge.x]));
                vel[edge.x].addInPlace(g.scale(dt));
                pos[edge.x].addInPlace(vel[edge.x].scale(dt));
            }

            if (!isFixed[edge.y]) {
                vel[edge.y].subtractInPlace(force.scale(dt / mass[edge.y]));
                vel[edge.y].addInPlace(g.scale(dt));
                pos[edge.y].addInPlace(vel[edge.y].scale(dt));
            }
        }
    }

}
