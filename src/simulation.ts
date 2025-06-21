import { Vector3 } from "@babylonjs/core";

export class Particle {
    pos: Vector3;
    vel: Vector3;
    mass: number;

    constructor(pos: Vector3, vel: Vector3, mass: number) {
        this.pos = pos.clone();
        this.vel = vel.clone();
        this.mass = mass;
    }

    update(deltaTime: number, gravity: Vector3) {
        const acceleration = gravity.scale(1 / this.mass);
        this.vel.addInPlace(acceleration.scale(deltaTime));
        this.pos.addInPlace(this.vel.scale(deltaTime));
        
        if (this.pos.y < -5) {
            this.pos.y = -5;
            this.vel.y = -this.vel.y * 0.8;
        }
    }
}

export class ParticleSystem {
    particles: Particle[] = [];
    gravity: Vector3;

    constructor(gravity: Vector3 = new Vector3(0, -9.8, 0)) {
        this.gravity = gravity;
    }

    addParticle(pos: Vector3, vel: Vector3, mass: number = 1.0) {
        const particle = new Particle(pos, vel, mass);
        this.particles.push(particle);
        return particle;
    }

    createRandomParticles(count: number) {
        for (let i = 0; i < count; i++) {
            const pos = new Vector3(
                (Math.random() - 0.5) * 10,
                Math.random() * 5 + 2,
                (Math.random() - 0.5) * 10
            );
            const vel = new Vector3(
                (Math.random() - 0.5) * 2,
                Math.random() * 2,
                (Math.random() - 0.5) * 2
            );
            const mass = 0.5 + Math.random() * 1.5;
            this.addParticle(pos, vel, mass);
        }
    }

    update(deltaTime: number) {
        for (const particle of this.particles) {
            particle.update(deltaTime, this.gravity);
        }
    }

    clear() {
        this.particles = [];
    }

    getParticlePositions(): Vector3[] {
        return this.particles.map(particle => particle.pos);
    }
}