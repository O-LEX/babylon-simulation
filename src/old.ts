import { Vector3 } from "@babylonjs/core";

class Node {
    position: Vector3;
    velocity: Vector3;
    mass: number;
    isFixed: boolean;

    constructor(position: Vector3, mass: number, isFixed: boolean = false) {
        this.position = position.clone();
        this.velocity = new Vector3(0, 0, 0);
        this.mass = mass;
        this.isFixed = isFixed;
    }
}

class Spring {
    node0: Node;
    node1: Node;
    restLength: number;
    stiffness: number;

    constructor(node0: Node, node1: Node, stiffness: number) {
        this.node0 = node0;
        this.node1 = node1;
        this.restLength = Vector3.Distance(node0.position, node1.position);
        this.stiffness = stiffness;
    }
}

class Geometry {
    nodes: Node[];
    springs: Spring[];

    constructor() {
        this.nodes = [];
        this.springs = [];
    }

    addNode(node: Node): void {
        this.nodes.push(node);
    }

    addSpring(spring: Spring): void {
        this.springs.push(spring);
    }
}

export class Solver {
    geometry: Geometry;
    gravity: Vector3 = new Vector3(0, -9.8, 0);

    get positions(): Vector3[] {
        return this.geometry.nodes.map(node => node.position);
    }

    constructor(geometry: Geometry) {
        this.geometry = geometry;
    }

    step(deltaTime: number) {
        const dt = deltaTime;
        const g = this.gravity;
        const springs = this.geometry.springs;

        for (const spring of springs) {
            const node0 = spring.node0;
            const node1 = spring.node1;

            const dir = node1.position.subtract(node0.position);
            const currentLength = dir.length();
            if (currentLength === 0) continue;

            const forceMagnitude = spring.stiffness * (currentLength - spring.restLength);
            const forceDir = dir.normalize();
            const force = forceDir.scale(forceMagnitude);

            if (!node0.isFixed) {
                node0.velocity.addInPlace(force.scale(dt / node0.mass));
                node0.velocity.addInPlace(g.scale(dt));
                node0.position.addInPlace(node0.velocity.scale(dt));
            }

            if (!node1.isFixed) {
                node1.velocity.subtractInPlace(force.scale(dt / node1.mass));
                node1.velocity.addInPlace(g.scale(dt));
                node1.position.addInPlace(node1.velocity.scale(dt));
            }
        }
    }
}

export class Cloth extends Geometry {
    width: number;
    height: number;
    segmentsX: number;
    segmentsY: number;

    constructor(width: number, height: number, segmentsX: number, segmentsY: number) {
        super();
        this.width = width;
        this.height = height;
        this.segmentsX = segmentsX;
        this.segmentsY = segmentsY;
        this.createCloth();
    }

    private createCloth(): void {
        const stepX = this.width / this.segmentsX;
        const stepY = this.height / this.segmentsY;

        // Create nodes in a grid pattern
        for (let y = 0; y <= this.segmentsY; y++) {
            for (let x = 0; x <= this.segmentsX; x++) {
                const position = new Vector3(
                    x * stepX - this.width / 2,
                    y * stepY,
                    0
                );
                const mass = 1.0;
                // fixed nodes
                const isFixed = (y === this.segmentsY && (x === 0 || x === this.segmentsX));
                const node = new Node(position, mass, isFixed);
                this.addNode(node);
            }
        }

        // Create horizontal springs
        for (let y = 0; y <= this.segmentsY; y++) {
            for (let x = 0; x < this.segmentsX; x++) {
                const index0 = y * (this.segmentsX + 1) + x;
                const index1 = y * (this.segmentsX + 1) + (x + 1);
                const spring = new Spring(this.nodes[index0], this.nodes[index1], 100);
                this.addSpring(spring);
            }
        }

        // Create vertical springs
        for (let y = 0; y < this.segmentsY; y++) {
            for (let x = 0; x <= this.segmentsX; x++) {
                const index0 = y * (this.segmentsX + 1) + x;
                const index1 = (y + 1) * (this.segmentsX + 1) + x;
                const spring = new Spring(this.nodes[index0], this.nodes[index1], 100);
                this.addSpring(spring);
            }
        }

        // Create diagonal springs for shear resistance
        for (let y = 0; y < this.segmentsY; y++) {
            for (let x = 0; x < this.segmentsX; x++) {
                const index0 = y * (this.segmentsX + 1) + x;
                const index1 = (y + 1) * (this.segmentsX + 1) + (x + 1);
                const index2 = y * (this.segmentsX + 1) + (x + 1);
                const index3 = (y + 1) * (this.segmentsX + 1) + x;
                
                // Cross diagonal springs
                const spring1 = new Spring(this.nodes[index0], this.nodes[index1], 50);
                const spring2 = new Spring(this.nodes[index2], this.nodes[index3], 50);
                this.addSpring(spring1);
                this.addSpring(spring2);
            }
        }
    }
}

// Factory function to create a cloth simulation
export function createCloth(width: number = 2, height: number = 2, segmentsX: number = 10, segmentsY: number = 10): Cloth {
    return new Cloth(width, height, segmentsX, segmentsY);
}
