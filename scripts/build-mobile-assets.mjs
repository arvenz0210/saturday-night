// Builds public/models/turntable.mobile.glb: weld + per-mesh simplification (only heavy parts,
// tight error so flat plates keep clean shading). Textures are resized/WebP'd afterwards with the CLI.
import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { simplifyPrimitive, weld } from "@gltf-transform/functions";
import { MeshoptSimplifier } from "meshoptimizer";

const [, , input, output, capArg = "14000"] = process.argv;
const cap = Number(capArg);
await MeshoptSimplifier.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.read(input);
await doc.transform(weld());
let before = 0, after = 0;
for (const mesh of doc.getRoot().listMeshes()) {
  for (const prim of mesh.listPrimitives()) {
    const tris = (prim.getIndices()?.getCount() ?? prim.getAttribute("POSITION").getCount()) / 3;
    before += tris;
    if (tris > cap) {
      // Large flat plates keep a tight error (visible shading), small detailed parts can be coarser.
      const pos = prim.getAttribute("POSITION");
      const [minX, minY, minZ] = pos.getMin([]);
      const [maxX, maxY, maxZ] = pos.getMax([]);
      const footprint = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
      // Plates and the platter show every shading kink: leave anything wider than 25 cm alone.
      if (footprint <= 0.25) {
        simplifyPrimitive(prim, { simplifier: MeshoptSimplifier, ratio: cap / tris, error: 0.003, lockBorder: true });
      }
    }
    after += (prim.getIndices()?.getCount() ?? prim.getAttribute("POSITION").getCount()) / 3;
  }
}
await io.write(output, doc);
console.log(`${input}: ${Math.round(before)} -> ${Math.round(after)} triangles (cap ${cap}/mesh)`);
