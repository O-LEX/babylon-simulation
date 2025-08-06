import { Vector3 } from "@babylonjs/core";

export interface Geometry {
    pos: Float32Array;            // [x0, y0, z0, x1, y1, z1, ...]
    masses: Float32Array;           // mass per vertex
    fixedVertices: Uint8Array;    // 0 or 1 per vertex
    edges: Uint32Array;           // [v0, v1, v2, v3, ...]
    stiffnesses: Float32Array;      // stiffness per edge
}

export function createCloth(
    width: number,
    height: number,
    resolutionX: number,
    resolutionY: number,
    stiffness: number = 100,
    mass: number = 1.0
): Geometry {
    const numCols = resolutionX + 1;
    const numRows = resolutionY + 1;
    const totalVertices = numCols * numRows;

    // positions: Float32Array (3 floats per vertex)
    const pos = new Float32Array(totalVertices * 3);

    // Fill positions
    for (let j = 0; j < numRows; j++) {
        for (let i = 0; i < numCols; i++) {
            const index = (j * numCols + i) * 3;
            pos[index] = (i / resolutionX) * width - width / 2;   // x
            pos[index + 1] = 5;                                   // y (fixed height)
            pos[index + 2] = (j / resolutionY) * height - height / 2; // z
        }
    }

    // Build edges: horizontal, vertical, diagonal (shear)
    const edgesList: number[] = [];

    // Horizontal edges
    for (let j = 0; j < numRows; j++) {
        for (let i = 0; i < resolutionX; i++) {
            const v0 = j * numCols + i;
            const v1 = v0 + 1;
            edgesList.push(v0, v1);
        }
    }

    // Vertical edges
    for (let j = 0; j < resolutionY; j++) {
        for (let i = 0; i < numCols; i++) {
            const v0 = j * numCols + i;
            const v1 = v0 + numCols;
            edgesList.push(v0, v1);
        }
    }

    // Diagonal (shear) edges
    for (let j = 0; j < resolutionY; j++) {
        for (let i = 0; i < resolutionX; i++) {
            const topLeft = j * numCols + i;
            const topRight = topLeft + 1;
            const bottomLeft = topLeft + numCols;
            const bottomRight = bottomLeft + 1;
            edgesList.push(topLeft, bottomRight); // \
            edgesList.push(topRight, bottomLeft); // /
        }
    }

    // Convert edgesList to Uint16Array
    const edges = new Uint32Array(edgesList);

    // Create stiffness array: one stiffness per edge
    const stiffnesses = new Float32Array(edges.length / 2);
    stiffnesses.fill(stiffness);

    // Create mass array: one mass per vertex
    const masses = new Float32Array(totalVertices);
    masses.fill(mass);

    // fixedVertices: Uint8Array, default 0 (movable)
    const fixedVertices = new Uint8Array(totalVertices);
    // Fix top-left and top-right corners
    fixedVertices[0] = 1;
    fixedVertices[resolutionX] = 1;

    return {
        pos,
        masses,
        fixedVertices,
        edges,
        stiffnesses
    };
}

export function createChain(length: number, resolution: number, stiffness: number = 100, mass: number = 1.0): Geometry {
    const numVertices = resolution;

    const pos = new Float32Array(numVertices * 3);

    for (let i = 0; i < numVertices; i++) {
        pos[i * 3] = (i / numVertices) * length; // x
        pos[i * 3 + 1] = 5;                     // y (fixed height)
        pos[i * 3 + 2] = 0;                     // z
    }

    const edgesList: number[] = [];
    
    // Create horizontal chain along x-axis
    for (let i = 0; i < numVertices - 1; i++) {
        edgesList.push(i, i + 1);
    }

    const edges = new Uint32Array(edgesList);

    const stiffnesses = new Float32Array(edges.length / 2);
    stiffnesses.fill(stiffness);

    const masses = new Float32Array(numVertices);
    masses.fill(mass);

    const fixedVertices = new Uint8Array(numVertices);
    fixedVertices[0] = 1; // Fix the first vertex

    const geometry = {
        pos,
        masses,
        fixedVertices,
        edges,
        stiffnesses,
    };

    return geometry;
}

export function createHighStiffnessRatioChain(): Geometry {
    const numVertices = 3;
    const pos = new Float32Array(numVertices * 3);
    for (let i = 0; i < numVertices; i++) {
        pos[i * 3] = i; // x
        pos[i * 3 + 1] = 5; // y (fixed height)
        pos[i * 3 + 2] = 0; // z
    }

    const edgesList: number[] = [];
    for (let i = 0; i < numVertices - 1; i++) {
        edgesList.push(i, i + 1);
    }

    const edges = new Uint32Array(edgesList);
    const stiffnesses = new Float32Array(edges.length / 2);
    stiffnesses.fill(1000000);
    stiffnesses[0] = 100;
    const masses = new Float32Array(numVertices);
    masses.fill(1.0);
    const fixedVertices = new Uint8Array(numVertices);
    fixedVertices[0] = 1;

    return {
        pos,
        masses,
        fixedVertices,
        edges,
        stiffnesses,
    };
}
