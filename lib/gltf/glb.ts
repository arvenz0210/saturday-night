// Minimal, dependency-free GLB (binary glTF 2.0) parser tailored for a PBR viewer.
// Produces interleaved vertex buffers (position, normal, uv, tangent), 32-bit
// indices, resolved materials, embedded images and flattened node instances.
import { mat4, vec3, type Mat4 } from "wgpu-matrix";

export type AlphaMode = "OPAQUE" | "MASK" | "BLEND";

export interface GlbImage {
  bytes: Uint8Array;
  mimeType: string;
  name: string;
}

export interface GlbSampler {
  magFilter?: number;
  minFilter?: number;
  wrapS: number;
  wrapT: number;
}

export interface GlbTextureRef {
  source: number; // image index
  sampler: number; // sampler index or -1
}

export interface GlbMaterial {
  name: string;
  baseColorFactor: [number, number, number, number];
  baseColorTexture: number; // texture index or -1
  metallicFactor: number;
  roughnessFactor: number;
  metallicRoughnessTexture: number;
  normalTexture: number;
  normalScale: number;
  occlusionTexture: number;
  occlusionStrength: number;
  emissiveTexture: number;
  emissiveFactor: [number, number, number];
  alphaMode: AlphaMode;
  alphaCutoff: number;
  doubleSided: boolean;
}

export interface GlbPrimitive {
  /** Interleaved: position(3) normal(3) uv(2) tangent(4) = 12 floats / 48 bytes per vertex. */
  vertices: Float32Array;
  vertexCount: number;
  indices: Uint32Array;
  material: number; // material index or -1
  hasTangents: boolean;
  min: [number, number, number];
  max: [number, number, number];
}

export interface GlbMesh {
  name: string;
  primitives: GlbPrimitive[];
}

export interface GlbInstance {
  mesh: number;
  node: number;
  name: string;
  worldMatrix: Mat4;
}

export interface GlbAsset {
  meshes: GlbMesh[];
  materials: GlbMaterial[];
  textures: GlbTextureRef[];
  images: GlbImage[];
  samplers: GlbSampler[];
  instances: GlbInstance[];
  /** World-space bounds of all instanced geometry. */
  min: [number, number, number];
  max: [number, number, number];
  triangleCount: number;
  extensionsUsed: string[];
}

export const VERTEX_STRIDE_FLOATS = 12;
export const VERTEX_STRIDE_BYTES = VERTEX_STRIDE_FLOATS * 4;

/* eslint-disable @typescript-eslint/no-explicit-any */
type Json = any;

const COMPONENT_SIZE: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

const TYPE_COMPONENTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};

export interface ParseGlbOptions {
  /** Instance every mesh with an identity matrix (raw accessor space), ignoring the node hierarchy. */
  ignoreNodeTransforms?: boolean;
}

export function parseGlb(buffer: ArrayBuffer, options: ParseGlbOptions = {}): GlbAsset {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x46546c67) {
    throw new Error("Not a GLB file (bad magic).");
  }
  const totalLength = view.getUint32(8, true);
  let offset = 12;
  let json: Json | undefined;
  let bin: Uint8Array | undefined;
  while (offset < totalLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    if (chunkType === 0x4e4f534a) {
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, chunkStart, chunkLength)));
    } else if (chunkType === 0x004e4942) {
      bin = new Uint8Array(buffer, chunkStart, chunkLength);
    }
    offset = chunkStart + chunkLength;
  }
  if (!json) throw new Error("GLB has no JSON chunk.");
  const extensionsUsed: string[] = json.extensionsUsed ?? [];
  const required: string[] = json.extensionsRequired ?? [];
  if (required.includes("KHR_draco_mesh_compression")) {
    throw new Error("This GLB uses Draco compression, which this viewer does not decode.");
  }

  const buffers: Uint8Array[] = (json.buffers ?? []).map((b: Json, i: number) => {
    if (b.uri) throw new Error(`External buffer ${i} not supported (GLB must be self-contained).`);
    if (!bin) throw new Error("GLB buffer references BIN chunk but none present.");
    return bin;
  });

  const bufferViewBytes = (index: number): Uint8Array => {
    const bv = json.bufferViews[index];
    const src = buffers[bv.buffer];
    return new Uint8Array(src.buffer, src.byteOffset + (bv.byteOffset ?? 0), bv.byteLength);
  };

  // Reads any accessor as a flat Float32Array (normalized ints are de-normalized).
  const readAccessorFloat = (index: number): { data: Float32Array; components: number; count: number } => {
    const acc = json.accessors[index];
    const components = TYPE_COMPONENTS[acc.type];
    const count: number = acc.count;
    const out = new Float32Array(count * components);
    if (acc.bufferView === undefined) {
      return { data: out, components, count }; // zero-filled (sparse-only accessors not supported)
    }
    const bv = json.bufferViews[acc.bufferView];
    const src = buffers[bv.buffer];
    const base = src.byteOffset + (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const compSize = COMPONENT_SIZE[acc.componentType];
    const stride: number = bv.byteStride ?? compSize * components;
    const dv = new DataView(src.buffer);
    const normalized = !!acc.normalized;
    // Fast path: float32 data can be copied with typed-array views instead of per-element reads.
    if (acc.componentType === 5126 && base % 4 === 0 && stride % 4 === 0) {
      const strideFloats = stride / 4;
      const view = new Float32Array(src.buffer, base, count === 0 ? 0 : (count - 1) * strideFloats + components);
      if (strideFloats === components) {
        out.set(view);
      } else {
        for (let i = 0; i < count; i++) {
          const o = i * strideFloats;
          for (let c = 0; c < components; c++) out[i * components + c] = view[o + c];
        }
      }
      return { data: out, components, count };
    }
    for (let i = 0; i < count; i++) {
      const elementBase = base + i * stride;
      for (let c = 0; c < components; c++) {
        const p = elementBase + c * compSize;
        let value: number;
        switch (acc.componentType) {
          case 5126: value = dv.getFloat32(p, true); break;
          case 5120: value = dv.getInt8(p); if (normalized) value = Math.max(value / 127, -1); break;
          case 5121: value = dv.getUint8(p); if (normalized) value /= 255; break;
          case 5122: value = dv.getInt16(p, true); if (normalized) value = Math.max(value / 32767, -1); break;
          case 5123: value = dv.getUint16(p, true); if (normalized) value /= 65535; break;
          case 5125: value = dv.getUint32(p, true); break;
          default: throw new Error(`Unsupported accessor componentType ${acc.componentType}`);
        }
        out[i * components + c] = value;
      }
    }
    return { data: out, components, count };
  };

  const readIndices = (index: number): Uint32Array => {
    const acc = json.accessors[index];
    const bv = json.bufferViews[acc.bufferView];
    const src = buffers[bv.buffer];
    const base = src.byteOffset + (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const count: number = acc.count;
    const out = new Uint32Array(count);
    const dv = new DataView(src.buffer);
    const compSize = COMPONENT_SIZE[acc.componentType];
    const stride: number = bv.byteStride ?? compSize;
    if (stride === compSize && base % compSize === 0) {
      if (acc.componentType === 5125) out.set(new Uint32Array(src.buffer, base, count));
      else if (acc.componentType === 5123) out.set(new Uint16Array(src.buffer, base, count));
      else out.set(new Uint8Array(src.buffer, base, count));
      return out;
    }
    for (let i = 0; i < count; i++) {
      const p = base + i * stride;
      out[i] =
        acc.componentType === 5125 ? dv.getUint32(p, true) :
        acc.componentType === 5123 ? dv.getUint16(p, true) :
        dv.getUint8(p);
    }
    return out;
  };

  // --- Images / samplers / textures -----------------------------------------------
  const images: GlbImage[] = (json.images ?? []).map((img: Json, i: number) => {
    if (img.bufferView === undefined) {
      throw new Error(`Image ${i} uses an external URI; only embedded images are supported.`);
    }
    return { bytes: bufferViewBytes(img.bufferView), mimeType: img.mimeType ?? "image/png", name: img.name ?? `image_${i}` };
  });
  const samplers: GlbSampler[] = (json.samplers ?? []).map((s: Json) => ({
    magFilter: s.magFilter,
    minFilter: s.minFilter,
    wrapS: s.wrapS ?? 10497,
    wrapT: s.wrapT ?? 10497,
  }));
  const textures: GlbTextureRef[] = (json.textures ?? []).map((t: Json) => {
    // Prefer the plain source; fall back to KHR_texture_basisu/webp sources if that's all there is.
    const source = t.source ?? t.extensions?.EXT_texture_webp?.source ?? -1;
    return { source, sampler: t.sampler ?? -1 };
  });

  // --- Materials ----------------------------------------------------------------------
  const texIndex = (info: Json | undefined): number => (info && typeof info.index === "number" ? info.index : -1);
  const materials: GlbMaterial[] = (json.materials ?? []).map((m: Json, i: number) => {
    const pbr = m.pbrMetallicRoughness ?? {};
    const specGloss = m.extensions?.KHR_materials_pbrSpecularGlossiness;
    let baseColorFactor: [number, number, number, number] = pbr.baseColorFactor ?? [1, 1, 1, 1];
    let baseColorTexture = texIndex(pbr.baseColorTexture);
    let metallicFactor: number = pbr.metallicFactor ?? 1;
    let roughnessFactor: number = pbr.roughnessFactor ?? 1;
    const metallicRoughnessTexture = texIndex(pbr.metallicRoughnessTexture);
    if (!m.pbrMetallicRoughness && specGloss) {
      // Rough conversion of the legacy spec/gloss model.
      baseColorFactor = specGloss.diffuseFactor ?? [1, 1, 1, 1];
      baseColorTexture = texIndex(specGloss.diffuseTexture);
      metallicFactor = 0;
      roughnessFactor = 1 - (specGloss.glossinessFactor ?? 1);
    }
    const emissiveStrength: number = m.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? 1;
    const emissive: [number, number, number] = m.emissiveFactor ?? [0, 0, 0];
    return {
      name: m.name ?? `material_${i}`,
      baseColorFactor,
      baseColorTexture,
      metallicFactor,
      roughnessFactor,
      metallicRoughnessTexture,
      normalTexture: texIndex(m.normalTexture),
      normalScale: m.normalTexture?.scale ?? 1,
      occlusionTexture: texIndex(m.occlusionTexture),
      occlusionStrength: m.occlusionTexture?.strength ?? 1,
      emissiveTexture: texIndex(m.emissiveTexture),
      emissiveFactor: [emissive[0] * emissiveStrength, emissive[1] * emissiveStrength, emissive[2] * emissiveStrength],
      alphaMode: (m.alphaMode ?? "OPAQUE") as AlphaMode,
      alphaCutoff: m.alphaCutoff ?? 0.5,
      doubleSided: !!m.doubleSided,
    };
  });

  // --- Meshes -----------------------------------------------------------------------------
  let triangleCount = 0;
  const meshes: GlbMesh[] = (json.meshes ?? []).map((mesh: Json, mi: number) => {
    const primitives: GlbPrimitive[] = [];
    for (const prim of mesh.primitives ?? []) {
      const mode: number = prim.mode ?? 4;
      if (mode !== 4) continue; // triangles only
      if (prim.extensions?.KHR_draco_mesh_compression) {
        throw new Error("Draco-compressed primitive encountered.");
      }
      const attrs = prim.attributes ?? {};
      if (attrs.POSITION === undefined) continue;
      const pos = readAccessorFloat(attrs.POSITION);
      const vertexCount = pos.count;
      let indices: Uint32Array;
      if (prim.indices !== undefined) {
        indices = readIndices(prim.indices);
      } else {
        indices = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) indices[i] = i;
      }
      const materialIndex: number = prim.material ?? -1;
      const material = materialIndex >= 0 ? materials[materialIndex] : undefined;

      const normals = attrs.NORMAL !== undefined ? readAccessorFloat(attrs.NORMAL).data : computeNormals(pos.data, indices);
      const uvs = attrs.TEXCOORD_0 !== undefined ? readAccessorFloat(attrs.TEXCOORD_0) : undefined;
      let uvData: Float32Array;
      if (uvs) {
        uvData = uvs.components === 2 ? uvs.data : repack(uvs.data, uvs.components, 2);
      } else {
        uvData = new Float32Array(vertexCount * 2);
      }
      let tangents: Float32Array | undefined;
      let hasTangents = false;
      if (attrs.TANGENT !== undefined) {
        const t = readAccessorFloat(attrs.TANGENT);
        tangents = t.components === 4 ? t.data : repack(t.data, t.components, 4, 1);
        hasTangents = true;
      } else if (material && material.normalTexture >= 0 && uvs) {
        tangents = computeTangents(pos.data, normals, uvData, indices);
        hasTangents = true;
      }

      const vertices = new Float32Array(vertexCount * VERTEX_STRIDE_FLOATS);
      const min: [number, number, number] = [Infinity, Infinity, Infinity];
      const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < vertexCount; i++) {
        const o = i * VERTEX_STRIDE_FLOATS;
        const px = pos.data[i * 3], py = pos.data[i * 3 + 1], pz = pos.data[i * 3 + 2];
        vertices[o] = px; vertices[o + 1] = py; vertices[o + 2] = pz;
        vertices[o + 3] = normals[i * 3]; vertices[o + 4] = normals[i * 3 + 1]; vertices[o + 5] = normals[i * 3 + 2];
        vertices[o + 6] = uvData[i * 2]; vertices[o + 7] = uvData[i * 2 + 1];
        if (tangents) {
          vertices[o + 8] = tangents[i * 4]; vertices[o + 9] = tangents[i * 4 + 1];
          vertices[o + 10] = tangents[i * 4 + 2]; vertices[o + 11] = tangents[i * 4 + 3] || 1;
        } else {
          vertices[o + 8] = 1; vertices[o + 9] = 0; vertices[o + 10] = 0; vertices[o + 11] = 1;
        }
        if (px < min[0]) min[0] = px; if (py < min[1]) min[1] = py; if (pz < min[2]) min[2] = pz;
        if (px > max[0]) max[0] = px; if (py > max[1]) max[1] = py; if (pz > max[2]) max[2] = pz;
      }
      triangleCount += indices.length / 3;
      primitives.push({ vertices, vertexCount, indices, material: materialIndex, hasTangents, min, max });
    }
    return { name: mesh.name ?? `mesh_${mi}`, primitives };
  });

  // --- Scene graph ------------------------------------------------------------------
  const instances: GlbInstance[] = [];
  const nodes: Json[] = json.nodes ?? [];
  const localMatrix = (node: Json): Mat4 => {
    if (node.matrix) return mat4.create(...(node.matrix as number[]));
    const t = node.translation ?? [0, 0, 0];
    const r = node.rotation ?? [0, 0, 0, 1];
    const s = node.scale ?? [1, 1, 1];
    const m = mat4.translation(t);
    mat4.multiply(m, mat4.fromQuat(r), m);
    mat4.scale(m, s, m);
    return m;
  };
  const visit = (nodeIndex: number, parent: Mat4) => {
    const node = nodes[nodeIndex];
    const world = options.ignoreNodeTransforms ? mat4.identity() : mat4.multiply(parent, localMatrix(node));
    if (node.mesh !== undefined) {
      instances.push({ mesh: node.mesh, node: nodeIndex, name: node.name ?? `node_${nodeIndex}`, worldMatrix: world });
    }
    for (const child of node.children ?? []) visit(child, world);
  };
  const sceneIndex: number = json.scene ?? 0;
  const roots: number[] = json.scenes?.[sceneIndex]?.nodes ?? nodes.map((_: Json, i: number) => i);
  for (const root of roots) visit(root, mat4.identity());

  // World bounds across instances (transform the 8 corners of each primitive box).
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const corner = vec3.create();
  for (const inst of instances) {
    for (const prim of meshes[inst.mesh].primitives) {
      for (let c = 0; c < 8; c++) {
        corner[0] = c & 1 ? prim.max[0] : prim.min[0];
        corner[1] = c & 2 ? prim.max[1] : prim.min[1];
        corner[2] = c & 4 ? prim.max[2] : prim.min[2];
        const w = vec3.transformMat4(corner, inst.worldMatrix);
        for (let k = 0; k < 3; k++) {
          if (w[k] < min[k]) min[k] = w[k];
          if (w[k] > max[k]) max[k] = w[k];
        }
      }
    }
  }

  return { meshes, materials, textures, images, samplers, instances, min, max, triangleCount, extensionsUsed };
}

function repack(src: Float32Array, srcComponents: number, dstComponents: number, fill = 0): Float32Array {
  const count = src.length / srcComponents;
  const out = new Float32Array(count * dstComponents);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < dstComponents; c++) {
      out[i * dstComponents + c] = c < srcComponents ? src[i * srcComponents + c] : fill;
    }
  }
  return out;
}

function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const count = positions.length / 3;
  const normals = new Float32Array(count * 3);
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i], b = indices[i + 1], c = indices[i + 2];
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
    const e1x = positions[b * 3] - ax, e1y = positions[b * 3 + 1] - ay, e1z = positions[b * 3 + 2] - az;
    const e2x = positions[c * 3] - ax, e2y = positions[c * 3 + 1] - ay, e2z = positions[c * 3 + 2] - az;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) {
      normals[v * 3] += nx; normals[v * 3 + 1] += ny; normals[v * 3 + 2] += nz;
    }
  }
  for (let i = 0; i < count; i++) {
    const x = normals[i * 3], y = normals[i * 3 + 1], z = normals[i * 3 + 2];
    const len = Math.hypot(x, y, z) || 1;
    normals[i * 3] = x / len; normals[i * 3 + 1] = y / len; normals[i * 3 + 2] = z / len;
  }
  return normals;
}

// Per-triangle tangent accumulation (Lengyel), Gram-Schmidt orthogonalized, with handedness in w.
function computeTangents(positions: Float32Array, normals: Float32Array, uvs: Float32Array, indices: Uint32Array): Float32Array {
  const count = positions.length / 3;
  const tan1 = new Float32Array(count * 3);
  const tan2 = new Float32Array(count * 3);
  for (let i = 0; i < indices.length; i += 3) {
    const i1 = indices[i], i2 = indices[i + 1], i3 = indices[i + 2];
    const x1 = positions[i2 * 3] - positions[i1 * 3];
    const y1 = positions[i2 * 3 + 1] - positions[i1 * 3 + 1];
    const z1 = positions[i2 * 3 + 2] - positions[i1 * 3 + 2];
    const x2 = positions[i3 * 3] - positions[i1 * 3];
    const y2 = positions[i3 * 3 + 1] - positions[i1 * 3 + 1];
    const z2 = positions[i3 * 3 + 2] - positions[i1 * 3 + 2];
    const s1 = uvs[i2 * 2] - uvs[i1 * 2];
    const t1 = uvs[i2 * 2 + 1] - uvs[i1 * 2 + 1];
    const s2 = uvs[i3 * 2] - uvs[i1 * 2];
    const t2 = uvs[i3 * 2 + 1] - uvs[i1 * 2 + 1];
    const det = s1 * t2 - s2 * t1;
    if (Math.abs(det) < 1e-12) continue;
    const r = 1 / det;
    const sx = (t2 * x1 - t1 * x2) * r, sy = (t2 * y1 - t1 * y2) * r, sz = (t2 * z1 - t1 * z2) * r;
    const tx = (s1 * x2 - s2 * x1) * r, ty = (s1 * y2 - s2 * y1) * r, tz = (s1 * z2 - s2 * z1) * r;
    for (const v of [i1, i2, i3]) {
      tan1[v * 3] += sx; tan1[v * 3 + 1] += sy; tan1[v * 3 + 2] += sz;
      tan2[v * 3] += tx; tan2[v * 3 + 1] += ty; tan2[v * 3 + 2] += tz;
    }
  }
  const out = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const nx = normals[i * 3], ny = normals[i * 3 + 1], nz = normals[i * 3 + 2];
    let tx = tan1[i * 3], ty = tan1[i * 3 + 1], tz = tan1[i * 3 + 2];
    const d = nx * tx + ny * ty + nz * tz;
    tx -= nx * d; ty -= ny * d; tz -= nz * d;
    let len = Math.hypot(tx, ty, tz);
    if (len < 1e-8) {
      // Degenerate UVs: pick any vector perpendicular to the normal.
      if (Math.abs(nx) < 0.9) { tx = 0; ty = -nz; tz = ny; } else { tx = nz; ty = 0; tz = -nx; }
      len = Math.hypot(tx, ty, tz) || 1;
    }
    tx /= len; ty /= len; tz /= len;
    // Handedness: sign of dot(cross(n, t), tan2)
    const cx = ny * tz - nz * ty, cy = nz * tx - nx * tz, cz = nx * ty - ny * tx;
    const w = cx * tan2[i * 3] + cy * tan2[i * 3 + 1] + cz * tan2[i * 3 + 2] < 0 ? -1 : 1;
    out[i * 4] = tx; out[i * 4 + 1] = ty; out[i * 4 + 2] = tz; out[i * 4 + 3] = w;
  }
  return out;
}
