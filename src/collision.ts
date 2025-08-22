import { Vector3 } from "@babylonjs/core";
import { udcf } from "./math/udcf";

export function Collision(pos: Float32Array, refPos: Float32Array, triangles: Uint32Array, fixedVertices: Uint8Array) {
    const numTriangles = triangles.length / 3;
    const numItr = 10;

    for (let itr = 0; itr < numItr; itr++) {
        for (let i = 0; i < numTriangles; i++) {
            for (let j = i + 1; j < numTriangles; j++) {
                const id_p0 = triangles[i * 3];
                let id_p1 = triangles[i * 3 + 1];
                let id_p2 = triangles[i * 3 + 2];
                const id_q0 = triangles[j * 3];
                let id_q1 = triangles[j * 3 + 1];
                let id_q2 = triangles[j * 3 + 2];

                // Use reference shape to decide orientation flip
                const ref_p0 = new Vector3(refPos[id_p0 * 3], refPos[id_p0 * 3 + 1], refPos[id_p0 * 3 + 2]);
                const ref_p1 = new Vector3(refPos[id_p1 * 3], refPos[id_p1 * 3 + 1], refPos[id_p1 * 3 + 2]);
                const ref_p2 = new Vector3(refPos[id_p2 * 3], refPos[id_p2 * 3 + 1], refPos[id_p2 * 3 + 2]);
                const ref_q0 = new Vector3(refPos[id_q0 * 3], refPos[id_q0 * 3 + 1], refPos[id_q0 * 3 + 2]);
                const ref_q1 = new Vector3(refPos[id_q1 * 3], refPos[id_q1 * 3 + 1], refPos[id_q1 * 3 + 2]);
                const ref_q2 = new Vector3(refPos[id_q2 * 3], refPos[id_q2 * 3 + 1], refPos[id_q2 * 3 + 2]);
                const np = Vector3.Cross(ref_p1.subtract(ref_p0), ref_p2.subtract(ref_p0));
                const cp = ref_p0.add(ref_p1).add(ref_p2).scale(1 / 3);
                const nq = Vector3.Cross(ref_q1.subtract(ref_q0), ref_q2.subtract(ref_q0));
                const cq = ref_q0.add(ref_q1).add(ref_q2).scale(1 / 3);
                const v = cq.subtract(cp);
                if (Vector3.Dot(np, v) < 0) {
                    const temp = id_p1;
                    id_p1 = id_p2;
                    id_p2 = temp;
                }
                if (Vector3.Dot(nq, v) > 0) {
                    const temp = id_q1;
                    id_q1 = id_q2;
                    id_q2 = temp;
                }

                const p0 = new Vector3(pos[id_p0 * 3], pos[id_p0 * 3 + 1], pos[id_p0 * 3 + 2]);
                const p1 = new Vector3(pos[id_p1 * 3], pos[id_p1 * 3 + 1], pos[id_p1 * 3 + 2]);
                const p2 = new Vector3(pos[id_p2 * 3], pos[id_p2 * 3 + 1], pos[id_p2 * 3 + 2]);
                const q0 = new Vector3(pos[id_q0 * 3], pos[id_q0 * 3 + 1], pos[id_q0 * 3 + 2]);
                const q1 = new Vector3(pos[id_q1 * 3], pos[id_q1 * 3 + 1], pos[id_q1 * 3 + 2]);
                const q2 = new Vector3(pos[id_q2 * 3], pos[id_q2 * 3 + 1], pos[id_q2 * 3 + 2]);

                const { C, dC } = udcf([p0, p1, p2], [q0, q1, q2]);

                const ids_p = [id_p0, id_p1, id_p2];
                for (let k = 0; k < 3; k++) {
                    if (fixedVertices[ids_p[k]]) continue;
                    const id = ids_p[k] * 3;
                    pos[id + 0] += dC.x * C;
                    pos[id + 1] += dC.y * C;
                    pos[id + 2] += dC.z * C;
                }
                const ids_q = [id_q0, id_q1, id_q2];
                for (let k = 0; k < 3; k++) {
                    if (fixedVertices[ids_q[k]]) continue;
                    const id = ids_q[k] * 3;
                    pos[id + 0] -= dC.x * C;
                    pos[id + 1] -= dC.y * C;
                    pos[id + 2] -= dC.z * C;
                }

            }
        }
    }
}