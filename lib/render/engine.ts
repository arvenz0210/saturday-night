// The viewer engine: owns the vgpu context, loads the GLB(s), builds GPU resources and
// runs the frame loop (shadow pass -> HDR PBR pass -> bloom -> tone-mapped composite).
import {
  clock, draw, effect, frameLoop, geometry, init, sampler, surface, target, uniforms,
  type Draw, type DrawOptions, type FrameLoopHandle, type Gpu, type SharedUniforms, type Texture,
} from "vgpu";
import { orbitControls, perspectiveCamera } from "vgpu/scene";
import { mat4, vec3, type Mat4 } from "wgpu-matrix";
import { parseGlb, VERTEX_STRIDE_BYTES, VERTEX_STRIDE_FLOATS, type GlbAsset, type GlbMaterial } from "@/lib/gltf/glb";
import type { AttachmentSpec, DeckButtonId, ModelSpec } from "@/lib/models";
import { buildEnvironment } from "./environment";
import { isTap, rayLocalBox, rayPlaneY, screenRay, TweenQueue, type Ray } from "./interaction";
import { DeckAudio } from "./audio";
import { Blitter, createImageTexture, createSolidTexture, gltfSamplerDescriptor } from "./textures";
import pbrShader from "./shaders/pbr.wgsl";
import shadowShader from "./shaders/shadow.wgsl";
import bloomBrightShader from "./shaders/bloom-bright.wgsl";
import bloomBlurShader from "./shaders/bloom-blur.wgsl";
import postShader from "./shaders/post.wgsl";

export interface ViewerSettings {
  exposure: number;
  /** Rotation of the studio lighting rig around the model, radians. */
  envRotation: number;
  envIntensity: number;
  lightIntensity: number;
  bloom: number;
  autoRotate: boolean;
  spin: boolean;
  finish: "original" | "black";
}

export interface ViewerLayout {
  /** "studio": dark set, orbit camera. "table": white background, fixed top-down camera. */
  theme: "studio" | "table";
  /** Fixed camera pose (radians) for the table theme; distance is fitted to the viewport. */
  camera?: { yaw: number; pitch: number; fit: number };
  /** Key light direction in degrees (azimuth 0 = front, elevation 90 = zenith). */
  light?: { azimuth: number; elevation: number };
  /** Caught-shadow darkness on the white table (0..1). */
  shadowStrength?: number;
  /** "mobile" lowers resolution, shadow map size and frame rate for phones. */
  quality?: "high" | "mobile";
}

export const DESKTOP_LAYOUT: ViewerLayout = { theme: "studio" };
export const MOBILE_LAYOUT: ViewerLayout = {
  theme: "table",
  camera: { yaw: -0.05, pitch: 1.1, fit: 1.0 },
  light: { azimuth: -12, elevation: 58 },
  shadowStrength: 0.48,
};

export const DEFAULT_SETTINGS: ViewerSettings = {
  exposure: 1.0,
  envRotation: 0,
  envIntensity: 1.0,
  lightIntensity: 2.6,
  bloom: 0.35,
  autoRotate: false,
  spin: true,
  finish: "black",
};

export type ViewerPhase = "gpu" | "download" | "parse" | "environment" | "textures" | "geometry" | "compile" | "ready";

export interface ViewerProgress {
  phase: ViewerPhase;
  loaded?: number;
  total?: number;
  detail?: string;
}

export interface ViewerInfo {
  adapter: string;
  triangles: number;
  materials: number;
  textures: number;
  drawCalls: number;
  extensions: string[];
  spinningNodes: string[];
  hasTonearm: boolean;
}

export interface ViewerState {
  /** Platter motor running. */
  playing: boolean;
  /** Stylus on the record (or travelling there). */
  armDown: boolean;
  armMoving: boolean;
  /** Speed change as a fraction, e.g. 0.04 = +4 %. */
  pitch: number;
  audioEnabled: boolean;
  /** Position along the record, 0 = lead-in, 1 = run-out. */
  progress: number;
  /** Selected nominal speed. */
  speed: 33 | 45;
  /** Quartz lock: pitch forced to 0 %. */
  quartz: boolean;
}

export interface ViewerCallbacks {
  onProgress(progress: ViewerProgress): void;
  onReady(info: ViewerInfo): void;
  onError(error: Error): void;
  onStats?(stats: { fps: number }): void;
  onState?(state: ViewerState): void;
}

export interface ViewerHandle {
  update(settings: Partial<ViewerSettings>): void;
  resetCamera(): void;
  togglePlay(): void;
  toggleArm(): void;
  setPitch(pitch: number): void;
  /** Starts the Web Audio context; call from a user gesture. */
  enableAudio(): Promise<void>;
  /** Fills `out` with analyser magnitudes; false until audio is enabled. */
  spectrum(out: Uint8Array<ArrayBuffer>): boolean;
  dispose(): void;
}

// Material.flags bits (mirrors pbr.wgsl)
const HAS_BASE_COLOR_TEX = 1;
const HAS_METALLIC_ROUGHNESS_TEX = 2;
const HAS_NORMAL_TEX = 4;
const HAS_EMISSIVE_TEX = 8;
const HAS_OCCLUSION_TEX = 16;
const ALPHA_MASK = 32;
const IS_FLOOR = 64;
const HAS_TANGENTS = 128;
const RECOLORABLE = 256;
const IS_LED = 512;

const SHADOW_SIZE_HIGH = 2048;
const SHADOW_SIZE_MOBILE = 1024;
const BACKGROUND: [number, number, number] = [0.012, 0.012, 0.015];
const PLATTER_RPM = 33.33;
/**
 * Adaptive quality ladder, cheapest first. Render scale is applied to the HDR target
 * (upscaled in post), taps to the soft shadow, bloom toggles its three passes.
 */
const QUALITY_LADDER = [
  { scale: 0.5, taps: 4, bloom: false },
  { scale: 0.6, taps: 4, bloom: false },
  { scale: 0.7, taps: 8, bloom: false },
  { scale: 0.85, taps: 8, bloom: true },
  { scale: 1.0, taps: 8, bloom: true },
  { scale: 1.0, taps: 16, bloom: true },
];
const FPS_FLOOR = 30; // step down below this
const FPS_HEADROOM = 54; // step up above this

/** Press feedback: the button's own LED blinks off briefly. */
const PRESS_BLINK = 0.14; // seconds
/** A full LP side: the stylus travels lead-in to run-out in about 20 minutes at 33⅓. */
const SIDE_SECONDS = 20 * 60;

const VERTEX_LAYOUT = {
  stride: VERTEX_STRIDE_BYTES,
  attributes: {
    position: "float32x3",
    normal: "float32x3",
    uv: "float32x2",
    tangent: "float32x4",
  },
} as const;

interface MaterialBindings {
  material: {
    baseColorFactor: number[];
    emissiveFactor: number[];
    metallicFactor: number;
    roughnessFactor: number;
    normalScale: number;
    occlusionStrength: number;
    alphaCutoff: number;
    flags: number;
  };
  baseColorTex: Texture;
  metallicRoughnessTex: Texture;
  normalTex: Texture;
  emissiveTex: Texture;
  occlusionTex: Texture;
  materialSampler: GPUSampler;
}

interface Drawable {
  main: Draw;
  shadow?: Draw;
  blend: boolean;
  center: Float32Array;
  meshIndex?: number;
  /** Skipped by both passes while true. */
  hidden?: boolean;
}

interface InstanceState {
  /** Static placement (includes the tonearm pose only through `pose`). */
  base: Mat4;
  model: SharedUniforms<{ model: Float32Array; normalMatrix: Float32Array }>;
  spin: boolean;
  pivot: Float32Array;
  name: string;
  meshIndex: number;
  /** Local bounds for picking. */
  bounds: Bounds[];
  /** Extra per-frame transform applied before spin (tonearm swing, fader travel). */
  pose?: Mat4;
  /** Set when the user can tap or drag this instance. */
  role?: "tonearm" | "pitch";
}

interface Bounds {
  min: number[];
  max: number[];
}

export function startViewer(
  canvas: HTMLCanvasElement,
  model: ModelSpec,
  initialSettings: ViewerSettings,
  callbacks: ViewerCallbacks,
  layout: ViewerLayout = DESKTOP_LAYOUT,
): ViewerHandle {
  const table = layout.theme === "table";
  const mobileQuality = layout.quality === "mobile";
  const SHADOW_SIZE = mobileQuality ? SHADOW_SIZE_MOBILE : SHADOW_SIZE_HIGH;
  const settings: ViewerSettings = { ...initialSettings };
  let disposed = false;
  let gpu: Gpu | undefined;
  let loop: FrameLoopHandle | undefined;
  let controls: ReturnType<typeof orbitControls> | undefined;
  let resetCameraImpl: (() => void) | undefined;
  let togglePlayImpl: (() => void) | undefined;
  let toggleArmImpl: (() => void) | undefined;
  let setPitchImpl: ((pitch: number) => void) | undefined;
  let audio: DeckAudio | undefined;
  let audioStateDirty = false;
  const cleanups: Array<() => void> = [];

  const run = async () => {
    const t0 = performance.now();
    const mark = (label: string) => {
      const entry = { label, ms: Math.round(performance.now() - t0) };
      const w = window as unknown as { __viewerTimings?: Array<{ label: string; ms: number }> };
      (w.__viewerTimings ??= []).push(entry);
    };
    let lastPhase: ViewerPhase | undefined;
    const rawOnProgress = callbacks.onProgress.bind(callbacks);
    callbacks = {
      ...callbacks,
      onProgress(progress) {
        if (progress.phase !== lastPhase) {
          mark(`phase:${progress.phase}`);
          lastPhase = progress.phase;
        }
        rawOnProgress(progress);
      },
    };
    if (!("gpu" in navigator) || !navigator.gpu) {
      throw new Error("WebGPU no está disponible en este navegador. Usa Chrome 113+, Edge 113+ o Safari 26+.");
    }
    callbacks.onProgress({ phase: "gpu" });
    gpu = await init({ powerPreference: "high-performance" });
    if (disposed) return;
    const ctx = gpu;
    cleanups.push(ctx.onError((error) => console.error("[vgpu]", error)));
    const device = ctx.gpu;
    device.addEventListener("uncapturederror", (event) => console.error("[webgpu]", (event as GPUUncapturedErrorEvent).error.message));
    const adapterInfo = ctx.device.adapterInfo;
    const adapterName = adapterInfo
      ? [adapterInfo.description, adapterInfo.vendor, adapterInfo.architecture].filter(Boolean).join(" · ") || "GPU"
      : "GPU";

    // --- Download + parse every GLB ------------------------------------------------------------
    const attachmentSpecs = model.attachments ?? [];
    const loadAsset = async (url: string, label: string, options: { ignoreNodeTransforms?: boolean } = {}): Promise<GlbAsset> => {
      const bytes = await fetchWithProgress(url, (loaded, total) =>
        callbacks.onProgress({ phase: "download", loaded, total, detail: label }));
      mark(`fetched:${label}`);
      callbacks.onProgress({ phase: "parse", detail: label });
      await nextFrame();
      const parsed = parseGlb(bytes, options);
      mark(`parsed:${label}`);
      return parsed;
    };
    const asset = await loadAsset(mobileQuality && model.mobileUrl ? model.mobileUrl : model.url, model.title);
    // Carve arm parts out of fused chassis meshes before anything else looks at the mesh list.
    const splitArmMeshes: number[] = [];
    for (const split of model.tonearm?.splitMeshes ?? []) {
      const created = splitMeshByRegion(asset, split.mesh, split.min, split.max);
      if (created >= 0) splitArmMeshes.push(created);
    }
    const armMeshes = new Set([...(model.tonearm?.meshes ?? []), ...splitArmMeshes]);
    const attachmentAssets: GlbAsset[] = [];
    for (const spec of attachmentSpecs) {
      attachmentAssets.push(await loadAsset(mobileQuality && spec.mobileUrl ? spec.mobileUrl : spec.url, spec.credit.source, { ignoreNodeTransforms: spec.ignoreNodeTransforms }));
    }
    if (disposed) return;

    // --- Lighting environment -------------------------------------------------------------------
    callbacks.onProgress({ phase: "environment" });
    const blitter = new Blitter(device);
    const env = await buildEnvironment(ctx, blitter, mobileQuality ? "mobile" : "high");
    mark("environment-built");
    if (disposed) return;

    // --- Render targets -------------------------------------------------------------------------------
    // The HDR target renders at `renderScale` of the canvas and is upscaled in post; the
    // adaptive controller moves that scale (plus shadow taps and bloom) to hold the frame rate.
    const canvasSurface = surface(ctx, canvas, { dpr: [1, 2] });
    let qualityLevel = mobileQuality ? 2 : QUALITY_LADDER.length - 1;
    let renderScale = QUALITY_LADDER[qualityLevel].scale;
    let bloomEnabled = QUALITY_LADDER[qualityLevel].bloom;
    const scaled = (w: number, h: number): [number, number] => [Math.max(1, Math.round(w * renderScale)), Math.max(1, Math.round(h * renderScale))];
    const [w0, h0] = scaled(...canvasSurface.size);
    // Table theme clears to alpha 0 so post composites the white background and the caught shadow.
    const clearColor: [number, number, number, number] = table ? [1, 1, 1, 0] : [...BACKGROUND, 1];
    const sceneTarget = target(ctx, { size: [w0, h0], format: "rgba16float", depth: "depth24plus", msaa: true, clearColor, label: "scene" });
    const quarter = (w: number, h: number): [number, number] => [Math.max(1, Math.floor(w / 4)), Math.max(1, Math.floor(h / 4))];
    const bloomBright = target(ctx, { size: quarter(w0, h0), format: "rgba16float", label: "bloom-bright" });
    const bloomA = target(ctx, { size: quarter(w0, h0), format: "rgba16float", label: "bloom-a" });
    const bloomB = target(ctx, { size: quarter(w0, h0), format: "rgba16float", label: "bloom-b" });
    const shadowTarget = target(ctx, { size: [SHADOW_SIZE, SHADOW_SIZE], format: "r8unorm", depth: "depth32float", label: "shadow" });
    const shadowSampler = sampler(ctx, { compare: "less", magFilter: "linear", minFilter: "linear" });
    const linearClamp = sampler(ctx, { addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", magFilter: "linear", minFilter: "linear" });

    // --- Shared uniforms --------------------------------------------------------------------------------
    const sceneUniforms = uniforms(ctx, {
      viewProjection: new Float32Array(16),
      lightViewProjection: new Float32Array(16),
      cameraPosition: [0, 0, 0],
      envRotation: 0,
      lightDirection: [0, 1, 0],
      lightIntensity: settings.lightIntensity,
      lightColor: [1, 0.96, 0.9],
      envIntensity: settings.envIntensity,
      shadowTexel: [1 / SHADOW_SIZE, 1 / SHADOW_SIZE],
      finish: settings.finish === "black" ? 1 : 0,
      specularMips: env.specularMips,
      backgroundColor: BACKGROUND,
      time: 0,
      floorMode: table ? 1 : 0,
      shadowSoftness: table ? 11 : 2.2,
      ledPower: settings.spin ? 1 : 0,
      hideBoxMin: [0, 0, 0],
      hideBoxActive: 0,
      hideBoxMax: [0, 0, 0],
      hideBox2Min: [0, 0, 0],
      hideBox2Max: [0, 0, 0],
      pressMinA: [0, 0, 0, 0], pressMaxA: [0, 0, 0, 0],
      pressMinB: [0, 0, 0, 0], pressMaxB: [0, 0, 0, 0],
      pressMinC: [0, 0, 0, 0], pressMaxC: [0, 0, 0, 0],
      pressMinD: [0, 0, 0, 0], pressMaxD: [0, 0, 0, 0],
      ledStates: [1, 1, 0, 0],
      shadowTaps: QUALITY_LADDER[qualityLevel].taps,
    });
    const lightUniforms = uniforms(ctx, { viewProjection: new Float32Array(16) });
    const sharedSceneBindings = {
      scene: sceneUniforms,
      shadowMap: shadowTarget.depth!,
      shadowSampler,
      envSpecular: env.specular,
      envIrradiance: env.irradiance,
      envSampler: env.sampler,
      brdfLut: env.brdfLut,
      lutSampler: env.lutSampler,
    };

    // --- Textures + materials (shared across assets) ------------------------------------------------
    const white = createSolidTexture(ctx, [1, 1, 1, 1], "white");
    const textureCache = new Map<string, Texture>();
    const neededTextures = [asset, ...attachmentAssets].reduce((n, a) => n + collectTextureUsage(a), 0);
    let uploaded = 0;
    const getTexture = async (source: GlbAsset, textureIndex: number, srgb: boolean): Promise<Texture> => {
      if (textureIndex < 0) return white;
      const ref = source.textures[textureIndex];
      if (!ref || ref.source < 0) return white;
      const image = source.images[ref.source];
      const key = `${image.bytes.byteOffset}:${image.bytes.byteLength}:${srgb ? "srgb" : "linear"}`;
      let tex = textureCache.get(key);
      if (!tex) {
        tex = await createImageTexture(ctx, blitter, image.bytes, image.mimeType, { srgb, label: image.name });
        textureCache.set(key, tex);
        uploaded++;
        callbacks.onProgress({ phase: "textures", loaded: uploaded, total: neededTextures, detail: image.name });
      }
      return tex;
    };
    const materialSamplerFor = (source: GlbAsset, mat: GlbMaterial): GPUSampler => {
      const texIndex = [mat.baseColorTexture, mat.normalTexture, mat.metallicRoughnessTexture].find((i) => i >= 0) ?? -1;
      const ref = texIndex >= 0 ? source.textures[texIndex] : undefined;
      const glbSampler = ref && ref.sampler >= 0 ? source.samplers[ref.sampler] : undefined;
      return sampler(ctx, gltfSamplerDescriptor(glbSampler));
    };
    const buildMaterials = async (source: GlbAsset, recolorable: boolean): Promise<MaterialBindings[]> => {
      const out: MaterialBindings[] = [];
      for (const mat of source.materials) {
        let flags = 0;
        if (mat.baseColorTexture >= 0) flags |= HAS_BASE_COLOR_TEX;
        if (mat.metallicRoughnessTexture >= 0) flags |= HAS_METALLIC_ROUGHNESS_TEX;
        if (mat.normalTexture >= 0) flags |= HAS_NORMAL_TEX;
        if (mat.emissiveTexture >= 0) flags |= HAS_EMISSIVE_TEX;
        if (mat.occlusionTexture >= 0) flags |= HAS_OCCLUSION_TEX;
        if (mat.alphaMode === "MASK") flags |= ALPHA_MASK;
        if (recolorable) flags |= RECOLORABLE;
        if (model.ledMaterialPattern?.test(mat.name)) flags |= IS_LED;
        out.push({
          material: {
            baseColorFactor: [...mat.baseColorFactor],
            emissiveFactor: [...mat.emissiveFactor],
            metallicFactor: mat.metallicFactor,
            roughnessFactor: mat.roughnessFactor,
            normalScale: mat.normalScale,
            occlusionStrength: mat.occlusionStrength,
            alphaCutoff: mat.alphaCutoff,
            flags,
          },
          baseColorTex: await getTexture(source, mat.baseColorTexture, true),
          metallicRoughnessTex: await getTexture(source, mat.metallicRoughnessTexture, false),
          normalTex: await getTexture(source, mat.normalTexture, false),
          emissiveTex: await getTexture(source, mat.emissiveTexture, true),
          occlusionTex: await getTexture(source, mat.occlusionTexture, false),
          materialSampler: materialSamplerFor(source, mat),
        });
      }
      return out;
    };
    const defaultMaterial: GlbMaterial = {
      name: "default", baseColorFactor: [0.8, 0.8, 0.8, 1], baseColorTexture: -1, metallicFactor: 0, roughnessFactor: 0.6,
      metallicRoughnessTexture: -1, normalTexture: -1, normalScale: 1, occlusionTexture: -1, occlusionStrength: 1,
      emissiveTexture: -1, emissiveFactor: [0, 0, 0], alphaMode: "OPAQUE", alphaCutoff: 0.5, doubleSided: false,
    };
    const defaultBindings: MaterialBindings = {
      material: {
        baseColorFactor: defaultMaterial.baseColorFactor, emissiveFactor: [0, 0, 0], metallicFactor: 0, roughnessFactor: 0.6,
        normalScale: 1, occlusionStrength: 1, alphaCutoff: 0.5, flags: model.recolorable ? RECOLORABLE : 0,
      },
      baseColorTex: white, metallicRoughnessTex: white, normalTex: white, emissiveTex: white, occlusionTex: white,
      materialSampler: sampler(ctx, gltfSamplerDescriptor()),
    };

    // --- Normalize the main model: Y up, centered on the floor, longest side = targetSize ---------
    const orientation = mat4.rotationY(model.rotateY ?? 0);
    if (model.upAxis === "z") mat4.rotateX(orientation, -Math.PI / 2, orientation);
    const oriented = transformBounds(asset.min, asset.max, orientation);
    const size = [oriented.max[0] - oriented.min[0], oriented.max[1] - oriented.min[1], oriented.max[2] - oriented.min[2]];
    const maxDim = Math.max(size[0], size[1], size[2]) || 1;
    const scale = model.targetSize / maxDim;
    const root = mat4.scaling([scale, scale, scale]);
    mat4.translate(root, [-(oriented.min[0] + oriented.max[0]) / 2, -oriented.min[1], -(oriented.min[2] + oriented.max[2]) / 2], root);
    mat4.multiply(root, orientation, root);
    const height = size[1] * scale;
    const footprint = Math.max(size[0], size[2]) * scale;
    const radius = 0.5 * Math.hypot(size[0], size[1], size[2]) * scale;
    const focus: [number, number, number] = [0, height * 0.42, 0];

    // World matrix of every main-model instance, then platter detection on those.
    const instanceWorld = asset.instances.map((inst) => mat4.multiply(root, inst.worldMatrix));
    const platterCandidates = new Set<number>();
    let platterTop = -Infinity;
    if (model.detectPlatter) {
      asset.instances.forEach((inst, index) => {
        const mesh = asset.meshes[inst.mesh];
        if (mesh.primitives.length === 0) return;
        const b = instanceBounds(mesh, instanceWorld[index]);
        const sx = b.max[0] - b.min[0], sy = b.max[1] - b.min[1], sz = b.max[2] - b.min[2];
        const square = Math.min(sx, sz) / Math.max(sx, sz);
        if (square > 0.95 && Math.max(sx, sz) > footprint * model.detectPlatter!.minFootprint && sy < 0.12 * Math.max(sx, sz)) {
          platterCandidates.add(index);
          platterTop = Math.max(platterTop, b.max[1]);
        }
      });
    }
    let platterPivot: Float32Array | undefined;
    if (platterCandidates.size > 0) {
      const acc = [0, 0, 0];
      for (const index of platterCandidates) {
        const c = instanceCenter(asset.meshes[asset.instances[index].mesh], instanceWorld[index]);
        acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2];
      }
      platterPivot = new Float32Array(acc.map((v) => v / platterCandidates.size));
    }

    // Attachment placement (before the tonearm, so the stylus can land on the record surface).
    const placements = attachmentSpecs.map((spec, i) => placeAttachment(attachmentAssets[i], spec, { scale, platterPivot, platterTop }));
    const surfaceTop = placements.reduce((top, p, i) => (p && attachmentSpecs[i].placeOnPlatter ? Math.max(top, p.top) : top), platterTop);

    // Tonearm: a pose is (yaw, tilt) around the bearing. `armPose(yaw, tilt)` builds the
    // world-space transform; the play pose puts the stylus on the lead-in groove and tilts
    // the arm down until the tip touches the record.
    const arm = model.tonearm && platterPivot ? model.tonearm : undefined;
    let armPlayYaw = 0;
    let armPlayTilt = 0;
    /** Yaw that puts the stylus at a given groove progress (0 = lead-in, 1 = run-out). */
    let armYawAt: (progress: number) => number = () => 0;
    let armPose: (yaw: number, tilt: number) => Mat4 = () => mat4.identity();
    if (arm && platterPivot) {
      const pivotW = vec3.transformMat4(vec3.create(...arm.pivot), root);
      const stylusW = vec3.transformMat4(vec3.create(...arm.stylus), root);
      armPlayYaw = solveTonearmAngle(pivotW, stylusW, platterPivot, arm.playRadius * scale);
      const leadIn = model.audio?.leadInRadius ?? arm.playRadius;
      const runOut = model.audio?.runOutRadius ?? arm.playRadius * 0.45;
      const yawLut = Array.from({ length: 33 }, (_, i) =>
        solveTonearmAngle(pivotW, stylusW, platterPivot!, (leadIn + (runOut - leadIn) * (i / 32)) * scale));
      armYawAt = (progress) => {
        const x = Math.max(0, Math.min(1, progress)) * 32;
        const i = Math.min(31, Math.floor(x));
        return yawLut[i] + (yawLut[i + 1] - yawLut[i]) * (x - i);
      };
      const yawMatrix = (yaw: number) => {
        const m = mat4.translation(pivotW);
        mat4.rotateY(m, yaw, m);
        mat4.translate(m, [-pivotW[0], -pivotW[1], -pivotW[2]], m);
        return m;
      };
      // Tilt axis: horizontal, perpendicular to the arm in its play position.
      const stylusPlay = vec3.transformMat4(stylusW, yawMatrix(armPlayYaw));
      const dir = vec3.normalize(vec3.create(stylusPlay[0] - pivotW[0], 0, stylusPlay[2] - pivotW[2]));
      const tiltAxis = vec3.normalize(vec3.cross(vec3.create(0, 1, 0), dir));
      const armLength = vec3.distance(pivotW, stylusPlay);
      // Leave a hair of air between the headshell underside and the vinyl.
      const clearance = 0.0012 * scale;
      const drop = stylusPlay[1] - (surfaceTop + clearance);
      // Sign of the tilt that lowers the stylus.
      let tiltSign = 1;
      if (Number.isFinite(surfaceTop) && Math.abs(drop) < armLength * 0.5) {
        const magnitude = Math.asin(drop / armLength);
        const probe = mat4.translation(pivotW);
        mat4.multiply(probe, mat4.axisRotation(tiltAxis, magnitude), probe);
        mat4.translate(probe, [-pivotW[0], -pivotW[1], -pivotW[2]], probe);
        tiltSign = vec3.transformMat4(stylusPlay, probe)[1] < stylusPlay[1] ? 1 : -1;
        armPlayTilt = magnitude;
      }
      armPose = (yaw, tilt) => {
        const m = mat4.translation(pivotW);
        mat4.multiply(m, mat4.axisRotation(tiltAxis, tilt * tiltSign), m);
        mat4.translate(m, [-pivotW[0], -pivotW[1], -pivotW[2]], m);
        return mat4.multiply(m, yawMatrix(yaw));
      };
      console.info(`[viewer] tonearm play pose: swing ${(armPlayYaw * 180 / Math.PI).toFixed(1)}°, tilt ${(armPlayTilt * 180 / Math.PI).toFixed(1)}°`);
    }

    // --- Geometry + draws --------------------------------------------------------------------------------
    callbacks.onProgress({ phase: "geometry" });
    await nextFrame();
    const instances: InstanceState[] = [];
    const drawables: Drawable[] = [];
    const spinningNodes: string[] = [];
    let triangleCount = 0;

    const buildAsset = async (source: GlbAsset, worlds: Mat4[], opts: { recolorable: boolean; spin(index: number, name: string): boolean; pivot?: Float32Array; label: string; role?(meshIndex: number): InstanceState["role"] }) => {
      const bindingsPerMaterial = await buildMaterials(source, opts.recolorable);
      if (disposed) return;
      for (const [index, inst] of source.instances.entries()) {
        const mesh = source.meshes[inst.mesh];
        if (mesh.primitives.length === 0) continue;
        const base = worlds[index];
        const spin = opts.spin(index, inst.name);
        const pivot = spin && opts.pivot ? opts.pivot : instanceCenter(mesh, base);
        const modelUniforms = uniforms(ctx, { model: new Float32Array(base), normalMatrix: normalMatrixOf(base) });
        instances.push({
          base, model: modelUniforms, spin, pivot, name: `${opts.label}/${inst.name}`, meshIndex: inst.mesh,
          bounds: mesh.primitives.map((prim) => ({ min: prim.min, max: prim.max })),
          role: opts.role?.(inst.mesh),
        });
        if (spin) spinningNodes.push(`${opts.label}/${inst.name}`);
        const mirrored = mat4.determinant(base) < 0;

        for (const prim of mesh.primitives) {
          const geo = geometry(ctx, {
            label: `${opts.label}:${mesh.name}`,
            buffers: [{ data: prim.vertices as Float32Array<ArrayBuffer>, ...VERTEX_LAYOUT }],
            indices: prim.indices as Uint32Array<ArrayBuffer>,
          });
          const glbMaterial = prim.material >= 0 ? source.materials[prim.material] : defaultMaterial;
          const bindings = prim.material >= 0 ? bindingsPerMaterial[prim.material] : defaultBindings;
          const flags = bindings.material.flags | (prim.hasTangents ? HAS_TANGENTS : 0);
          const blend = glbMaterial.alphaMode === "BLEND";
          const mainOptions: DrawOptions = {
            label: `pbr:${opts.label}:${mesh.name}:${glbMaterial.name}`,
            shader: pbrShader,
            geometry: geo,
            cull: glbMaterial.doubleSided ? "none" : "back",
            frontFace: mirrored ? "cw" : "ccw",
            ...(blend ? { blend: "alpha" as const, depth: { write: false } } : {}),
            set: { ...sharedSceneBindings, ...bindings, material: { ...bindings.material, flags }, model: modelUniforms },
          };
          const main = draw(ctx, mainOptions);
          const shadow = blend
            ? undefined
            : draw(ctx, {
                label: `shadow:${opts.label}:${mesh.name}`,
                shader: shadowShader,
                geometry: geo,
                cull: "none",
                frontFace: mirrored ? "cw" : "ccw",
                writeMask: [],
                depth: { bias: 2, biasSlopeScale: 2.5 },
                set: { light: lightUniforms, model: modelUniforms },
              });
          const localCenter = vec3.create((prim.min[0] + prim.max[0]) / 2, (prim.min[1] + prim.max[1]) / 2, (prim.min[2] + prim.max[2]) / 2);
          drawables.push({ main, shadow, blend, center: vec3.transformMat4(localCenter, base), meshIndex: source === asset ? inst.mesh : undefined });
          triangleCount += prim.indices.length / 3;
        }
      }
    };

    await buildAsset(asset, instanceWorld, {
      recolorable: model.recolorable,
      spin: (index, name) => platterCandidates.has(index) || (!!model.spinPattern && (model.spinPattern.test(name) || model.spinPattern.test(asset.meshes[asset.instances[index].mesh].name))),
      pivot: platterPivot,
      label: model.id,
      role: (meshIndex) => {
        if (arm && armMeshes.has(meshIndex)) return "tonearm";
        const pitch = model.controls?.pitch;
        if (pitch && (pitch.meshes.includes(meshIndex) || pitch.grabMeshes?.includes(meshIndex))) return "pitch";
        return undefined;
      },
    });
    if (disposed) return;

    // --- Attachments (e.g. the record on the platter) ------------------------------------------------
    for (const [i, spec] of attachmentSpecs.entries()) {
      const placement = placements[i];
      if (!placement) {
        console.warn(`[viewer] attachment "${spec.id}" skipped: no platter detected to place it on.`);
        continue;
      }
      await buildAsset(attachmentAssets[i], placement.worlds, { recolorable: false, spin: () => spec.spin, pivot: platterPivot, label: spec.id });
      if (disposed) return;
    }

    mark("geometry-built");
    // Record label decal: the track's cover art on a thin annulus riding the record.
    if (model.label && platterPivot && Number.isFinite(surfaceTop)) {
      try {
        const response = await fetch(model.label.image);
        if (!response.ok) throw new Error(`${response.status}`);
        const bytes = new Uint8Array(await response.arrayBuffer());
        const labelTexture = await createImageTexture(ctx, blitter, bytes, response.headers.get("content-type") ?? "image/jpeg", { srgb: true, label: "record-label" });
        const outer = (model.label.diameter / 2) * scale;
        const inner = (model.label.hole / 2) * scale;
        const labelGeometry = geometry(ctx, {
          label: "record-label",
          buffers: [{ data: annulusVertices(inner, outer, 96, labelTexture.size[1] / labelTexture.size[0]) as Float32Array<ArrayBuffer>, ...VERTEX_LAYOUT }],
          indices: annulusIndices(96),
        });
        const base = mat4.translation([platterPivot[0], surfaceTop + 0.0004 * scale, platterPivot[2]]);
        const labelModel = uniforms(ctx, { model: new Float32Array(base), normalMatrix: normalMatrixOf(base) });
        const labelBindings: MaterialBindings = {
          ...defaultBindings,
          material: { ...defaultBindings.material, baseColorFactor: [1, 1, 1, 1], roughnessFactor: 0.55, metallicFactor: 0, flags: HAS_BASE_COLOR_TEX },
          baseColorTex: labelTexture,
          materialSampler: sampler(ctx, gltfSamplerDescriptor({ wrapS: 33071, wrapT: 33071 })),
        };
        const labelDraw = draw(ctx, {
          label: "record-label",
          shader: pbrShader,
          geometry: labelGeometry,
          cull: "none",
          set: { ...sharedSceneBindings, ...labelBindings, model: labelModel },
        });
        instances.push({ base, model: labelModel, spin: true, pivot: platterPivot, name: "label", meshIndex: -1, bounds: [] });
        drawables.push({ main: labelDraw, shadow: undefined, blend: false, center: vec3.create(platterPivot[0], surfaceTop, platterPivot[2]) });
      } catch (error) {
        console.warn("[viewer] label art skipped:", error);
      }
    }

    // Audio: fetched now, decoded on the first user gesture.
    if (model.audio) {
      audio = new DeckAudio({
        url: model.audio.url,
        onEnded: () => console.info("[viewer] track finished; lift the arm and drop it again to replay"),
      });
      audio.loading.catch((error) => console.warn("[viewer] audio unavailable:", error));
    }

    // Floor: a large quad that receives the shadow and fades into the backdrop.
    const floorGeometry = geometry(ctx, {
      label: "floor",
      buffers: [{ data: floorVertices(6) as Float32Array<ArrayBuffer>, ...VERTEX_LAYOUT }],
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });
    const floorModel = uniforms(ctx, { model: new Float32Array(mat4.translation([0, -0.0005, 0])), normalMatrix: new Float32Array(mat4.identity()) });
    const floorDraw = draw(ctx, {
      label: "floor",
      shader: pbrShader,
      geometry: floorGeometry,
      cull: "back",
      set: {
        ...sharedSceneBindings,
        ...defaultBindings,
        material: { ...defaultBindings.material, flags: IS_FLOOR },
        model: floorModel,
      },
    });

    const postParams = (aspect: number) => ({
      exposure: settings.exposure,
      bloomStrength: bloomEnabled ? settings.bloom : 0,
      vignette: table ? 0 : 0.35,
      aspect,
      background: [0.925, 0.925, 0.925],
      shadowStrength: layout.shadowStrength ?? 0.3,
    });
    let fitCamera: ((aspect: number) => void) | undefined = undefined;

    // --- Post processing ----------------------------------------------------------------------------------
    const bright = effect(ctx, bloomBrightShader, {
      label: "bloom-bright",
      set: { params: { texel: [...sceneTarget.texelSize], threshold: 1.0, knee: 0.6 }, src: sceneTarget, samp: linearClamp },
    });
    const blurH = effect(ctx, bloomBlurShader, {
      label: "bloom-blur-h",
      set: { params: { direction: [bloomBright.texelSize[0], 0] }, src: bloomBright, samp: linearClamp },
    });
    const blurV = effect(ctx, bloomBlurShader, {
      label: "bloom-blur-v",
      set: { params: { direction: [0, bloomBright.texelSize[1]] }, src: bloomA, samp: linearClamp },
    });
    const post = effect(ctx, postShader, {
      label: "post",
      set: {
        params: postParams(w0 / h0),
        scene: sceneTarget,
        bloom: bloomB,
        samp: linearClamp,
      },
    });

    // --- Camera ---------------------------------------------------------------------------------------------
    const fov = 30;
    const camera = perspectiveCamera({ fov, aspect: w0 / h0, near: 0.02, far: 50, position: [0, focus[1], 3], target: focus });
    // Table theme: no pointer input reaches the camera; it stays where the fit puts it.
    controls = orbitControls(camera, {
      ...(table ? {} : { element: canvas }),
      target: focus,
      damping: 0.12,
      rotateSpeed: 0.005,
      zoomSpeed: 1,
      distance: { min: radius * 0.6, max: radius * 12 },
      pitch: { min: 0.02, max: 1.45 },
    });
    // The album floats over the lower part of the screen: aim below the deck so it sits high.
    const tableFocus: [number, number, number] = [0, height * 0.5 - radius * 0.55, 0];
    fitCamera = table
      ? (aspect) => {
          // Distance so the deck's bounding sphere fits the narrower field of view.
          const vHalf = (fov * Math.PI) / 360;
          const hHalf = Math.atan(Math.tan(vHalf) * aspect);
          const half = Math.min(vHalf, hHalf);
          const pose = layout.camera ?? MOBILE_LAYOUT.camera!;
          // Portrait screens are width-bound and show the deck at a shallow angle: give them more room.
          const portrait = aspect < 1 ? 1.14 : 1;
          const distance = (radius * 0.92 * pose.fit * portrait) / Math.sin(half);
          controls!.set({ yaw: pose.yaw, pitch: pose.pitch, distance, target: tableFocus });
        }
      : undefined;
    const applyCameraPose = () => {
      if (fitCamera) fitCamera(camera.aspect);
      else controls!.set({ yaw: model.camera.yaw, pitch: model.camera.pitch, distance: radius * model.camera.distance, target: focus });
    };
    applyCameraPose();
    resetCameraImpl = applyCameraPose;

    let lastInteraction = -Infinity;
    const markInteraction = () => { lastInteraction = performance.now(); };
    canvas.addEventListener("pointerdown", markInteraction);
    canvas.addEventListener("wheel", markInteraction, { passive: true });
    cleanups.push(() => {
      canvas.removeEventListener("pointerdown", markInteraction);
      canvas.removeEventListener("wheel", markInteraction);
    });

    // --- Resize ----------------------------------------------------------------------------------------------
    const resizeTargets = (width: number, height: number) => {
      const [w, h] = scaled(width, height);
      sceneTarget.resize([w, h]);
      const q = quarter(w, h);
      bloomBright.resize(q);
      bloomA.resize(q);
      bloomB.resize(q);
      bright.set({ params: { texel: [...sceneTarget.texelSize], threshold: 1.0, knee: 0.6 } });
      blurH.set({ params: { direction: [bloomBright.texelSize[0], 0] } });
      blurV.set({ params: { direction: [0, bloomBright.texelSize[1]] } });
      post.set({ params: postParams(width / height) });
    };
    const applyQuality = (level: number) => {
      qualityLevel = Math.max(0, Math.min(QUALITY_LADDER.length - 1, level));
      const q = QUALITY_LADDER[qualityLevel];
      renderScale = q.scale;
      bloomEnabled = q.bloom;
      sceneUniforms.set({ shadowTaps: q.taps });
      resizeTargets(...canvasSurface.size);
      console.info(`[viewer] quality level ${qualityLevel}: scale ${q.scale}, ${q.taps} shadow taps, bloom ${q.bloom ? "on" : "off"}`);
    };
    cleanups.push(canvasSurface.onResize(({ width, height }) => {
      resizeTargets(width, height);
      camera.set({ aspect: width / height });
      if (fitCamera) fitCamera(width / height);
    }));

    // --- Pre-warm pipelines so the first frame does not hitch ------------------------------------------
    callbacks.onProgress({ phase: "compile" });
    await Promise.all([
      ...drawables.map((d) => d.main.compile(sceneTarget)),
      ...drawables.filter((d) => d.shadow).map((d) => d.shadow!.compile(shadowTarget)),
      floorDraw.compile(sceneTarget),
      bright.compile(bloomBright),
      blurH.compile(bloomA),
      blurV.compile(bloomB),
      post.compile({ colors: [canvasSurface.format] }),
    ]);
    mark("compiled");
    if (disposed) return;

    // --- Frame loop -----------------------------------------------------------------------------------------
    const time = clock(ctx);
    const lightView = mat4.create();
    const lightProjection = mat4.ortho(-radius, radius, -radius, radius, radius * 0.5, radius * 4.5);
    const lightViewProjection = mat4.create();
    const spinMatrix = mat4.create();
    let spinAngle = 0;
    let ledPower = settings.spin ? 1 : 0;

    // --- Deck state: motor, tonearm, pitch --------------------------------------------------------------
    const deck = {
      running: settings.spin,
      omega: settings.spin ? (PLATTER_RPM / 60) * Math.PI * 2 : 0,
      pitch: 0,
      speed: 33 as 33 | 45,
      quartz: false,
      armDown: !!arm,
      armValues: { yaw: arm ? armPlayYaw : 0, tilt: arm ? armPlayTilt : 0 },
    };
    const armTweens = new TweenQueue<"yaw" | "tilt">(deck.armValues);
    const armInstances = instances.filter((i) => i.role === "tonearm");
    const hideRegions = (arm?.hideRegions ?? []).slice(0, 2).map((r) => transformBounds(r.min, r.max, root));
    if (hideRegions[0]) sceneUniforms.set({ hideBoxMin: hideRegions[0].min, hideBoxMax: hideRegions[0].max, hideBoxActive: 1 });
    if (hideRegions[1]) sceneUniforms.set({ hideBox2Min: hideRegions[1].min, hideBox2Max: hideRegions[1].max });
    const restOnlyDrawables = drawables.filter((d) => d.meshIndex !== undefined && arm?.hideAwayFromRest?.includes(d.meshIndex));
    const pitchSpec = model.controls?.pitch;
    const pitchSpecMoves = (meshIndex: number) => !!pitchSpec?.meshes.includes(meshIndex);
    const pitchInstances = instances.filter((i) => i.role === "pitch" && pitchSpecMoves(i.meshIndex));
    const pitchAxis = pitchSpec ? vec3.normalize(vec3.create(...pitchSpec.axis)) : vec3.create(0, 0, 1);
    let progress = 0;
    const emitState = () => callbacks.onState?.({ playing: deck.running, armDown: deck.armDown, armMoving: armTweens.busy, pitch: deck.pitch, audioEnabled: !!audio?.enabled, progress, speed: deck.speed, quartz: deck.quartz });
    const applyArmPose = () => {
      const pose = armPose(deck.armValues.yaw, deck.armValues.tilt);
      for (const inst of armInstances) inst.pose = pose;
    };
    applyArmPose();
    if (new URLSearchParams(location.search).has("debug")) {
      (window as unknown as { __instances?: unknown }).__instances = instances.map((i) => ({ name: i.name, mesh: i.meshIndex, role: i.role, pose: !!i.pose, spin: i.spin }));
    }
    const applyPitchPose = () => {
      if (!pitchSpec) return;
      const offset = (deck.pitch / pitchSpec.maxPitch) * pitchSpec.halfTravel * scale;
      const pose = mat4.translation(vec3.scale(pitchAxis, offset));
      for (const inst of pitchInstances) inst.pose = pose;
    };
    togglePlayImpl = () => pressButton("startStop");
    toggleArmImpl = () => {
      if (!arm) return;
      deck.armDown = !deck.armDown;
      if (deck.armDown && audio && progress >= 0.999) {
        // Dropping the arm after the run-out starts the side again.
        audio.seek(0);
        progress = 0;
      }
      // Lift, travel, lower: the cue lever choreography.
      armTweens.set(deck.armDown
        ? [{ prop: "tilt", to: 0, duration: 0.35 }, { prop: "yaw", to: armYawAt(progress), duration: 1.1 }, { prop: "tilt", to: armPlayTilt, duration: 0.5 }]
        : [{ prop: "tilt", to: 0, duration: 0.35 }, { prop: "yaw", to: 0, duration: 1.1 }, { prop: "tilt", to: 0, duration: 0.2 }]);
      emitState();
    };
    setPitchImpl = (pitch) => {
      if (!pitchSpec) return;
      deck.pitch = Math.max(-pitchSpec.maxPitch, Math.min(pitchSpec.maxPitch, pitch));
      applyPitchPose();
      emitState();
    };
    // Instances whose world matrix changed this frame need their uniforms rewritten.
    const dirtyStatic = new Set<InstanceState>(instances);
    const writeInstance = (inst: InstanceState, spin: Mat4 | undefined) => {
      let world: Mat4 = inst.base;
      if (inst.pose) world = mat4.multiply(inst.pose, world);
      if (spin) world = mat4.multiply(spin, world);
      inst.model.set({ model: new Float32Array(world), normalMatrix: normalMatrixOf(world) });
    };
    /** Current world matrix (pose + spin) of an instance, for picking. */
    const currentWorld = (inst: InstanceState): Mat4 => {
      let world: Mat4 = inst.base;
      if (inst.pose) world = mat4.multiply(inst.pose, world);
      if (inst.spin) {
        mat4.translation(inst.pivot, spinMatrix);
        mat4.rotateY(spinMatrix, spinAngle, spinMatrix);
        mat4.translate(spinMatrix, [-inst.pivot[0], -inst.pivot[1], -inst.pivot[2]], spinMatrix);
        world = mat4.multiply(spinMatrix, world);
      }
      return world;
    };

    // --- Gestures: tap the arm, tap START/STOP, drag the pitch fader ---------------------------------
    // Physical buttons: world-space boxes, press animation state, LED multipliers.
    const buttonSlots = ["A", "B", "C", "D"] as const;
    const buttons = (model.controls?.buttons ?? []).slice(0, 4).map((b, i) => ({
      id: b.id,
      slot: buttonSlots[i],
      bounds: transformBounds(b.min, b.max, root),
      pressedAt: -Infinity,
    }));
    for (const b of buttons) {
      sceneUniforms.set({ [`pressMin${b.slot}`]: [...b.bounds.min, 0], [`pressMax${b.slot}`]: [...b.bounds.max, 0] });
    }
    const identity = mat4.identity();
    const ledStateOf = (id: DeckButtonId) =>
      id === "speed33" ? (deck.speed === 33 ? 1 : 0.08)
        : id === "speed45" ? (deck.speed === 45 ? 1 : 0.08)
        : id === "quartz" ? (deck.quartz ? 1 : 0.05)
        : 1;
    const updateLeds = (now = performance.now()) => {
      const states = [1, 1, 1, 1];
      for (const [i, b] of buttons.entries()) {
        const blinking = now - b.pressedAt < PRESS_BLINK * 1000;
        states[i] = blinking ? 0.15 : ledStateOf(b.id);
      }
      sceneUniforms.set({ ledStates: states });
    };
    updateLeds();
    const pressButton = (id: DeckButtonId) => {
      const b = buttons.find((x) => x.id === id);
      if (b) b.pressedAt = performance.now();
      if (id === "startStop") { deck.running = !deck.running; settings.spin = deck.running; }
      else if (id === "speed33") deck.speed = 33;
      else if (id === "speed45") deck.speed = 45;
      else if (id === "quartz") deck.quartz = !deck.quartz;
      updateLeds();
      emitState();
    };
    const rayAt = (event: PointerEvent): Ray => {
      const rect = canvas.getBoundingClientRect();
      return screenRay(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height, camera.viewProjection, camera.worldPosition);
    };
    type Hit = { kind: "tonearm" | "pitch" | "button" | "other"; t: number; inst?: InstanceState; button?: DeckButtonId };
    // Bounding boxes of the chassis swallow the small parts, so interactive hits win
    // whenever the ray touches one; plain geometry is only reported when nothing else is.
    const pick = (ray: Ray, interactiveOnly: boolean): Hit | undefined => {
      let best: Hit | undefined;
      let bestOther: Hit | undefined;
      for (const inst of instances) {
        if (interactiveOnly && !inst.role) continue;
        const inverse = mat4.invert(currentWorld(inst));
        const pad = inst.role === "pitch" ? 0.012 : inst.role === "tonearm" ? 0.006 : 0;
        for (const b of inst.bounds) {
          const t = rayLocalBox(ray, inverse, b.min, b.max, pad);
          if (t === null) continue;
          if (inst.role) {
            if (!best || t < best.t) best = { kind: inst.role, t, inst };
          } else if (!bestOther || t < bestOther.t) {
            bestOther = { kind: "other", t, inst };
          }
        }
      }
      for (const b of buttons) {
        const t = rayLocalBox(ray, identity, b.bounds.min, b.bounds.max, 0.004 * scale);
        if (t !== null && (!best || t < best.t)) best = { kind: "button", t, button: b.id };
      }
      return best ?? bestOther;
    };
    let pointerDown: { x: number; y: number; time: number; id: number } | undefined;
    let pitchDrag: { grabOffset: number; height: number } | undefined;
    // Table theme: the camera has no orbit element, so gestures are handled here.
    const touches = new Map<number, { x: number; y: number }>();
    let orbitDrag: { x: number; y: number; yaw: number; pitch: number } | undefined;
    let pinch: { distance: number; angle: number; camDistance: number; yaw: number } | undefined;
    const beginPinch = () => {
      const [a, b] = [...touches.values()];
      pinch = { distance: Math.hypot(b.x - a.x, b.y - a.y), angle: Math.atan2(b.y - a.y, b.x - a.x), camDistance: controls!.distance, yaw: controls!.yaw };
      orbitDrag = undefined;
      pitchDrag = undefined;
      pointerDown = undefined;
    };
    const interactiveSince = performance.now() + 400; // settle time: ignore synthetic events right after load
    const onPointerDown = (event: PointerEvent) => {
      if (event.button !== 0 || performance.now() < interactiveSince) return;
      if (table) {
        touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
        try { canvas.setPointerCapture(event.pointerId); } catch { /* synthetic or already-released pointer */ }
        if (touches.size === 2) { beginPinch(); return; }
        if (touches.size > 2) return;
      }
      pointerDown = { x: event.clientX, y: event.clientY, time: performance.now(), id: event.pointerId };
      const hit = pick(rayAt(event), true);
      if (table && !hit && !pick(rayAt(event), false)) {
        // Finger on the table, not the deck: orbit.
        orbitDrag = { x: event.clientX, y: event.clientY, yaw: controls!.yaw, pitch: controls!.pitch };
        return;
      }
      if (hit?.kind === "pitch" && hit.inst && pitchSpec) {
        // Start dragging the fader: orbit controls must not see this gesture.
        event.stopImmediatePropagation();
        event.preventDefault();
        const knobCenter = vec3.transformMat4(instanceCenterLocal(hit.inst), currentWorld(hit.inst));
        const plane = rayPlaneY(rayAt(event), knobCenter[1]);
        const along = plane ? vec3.dot(vec3.sub(plane, knobCenter), pitchAxis) : 0;
        // Grabbing the slot (not the knob) jumps the knob to the pointer.
        if (!pitchSpecMoves(hit.inst.meshIndex)) pitchDrag = { grabOffset: 0, height: knobCenter[1] };
        else pitchDrag = { grabOffset: along, height: knobCenter[1] };
        canvas.setPointerCapture(event.pointerId);
        canvas.style.cursor = "grabbing";
      }
    };
    const onPointerMove = (event: PointerEvent) => {
      if (table && touches.has(event.pointerId)) touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (pinch && touches.size >= 2) {
        const [a, b] = [...touches.values()];
        const distance = Math.hypot(b.x - a.x, b.y - a.y);
        const angle = Math.atan2(b.y - a.y, b.x - a.x);
        controls!.set({
          distance: pinch.camDistance * (pinch.distance / Math.max(distance, 1)),
          yaw: pinch.yaw + (angle - pinch.angle),
        });
        return;
      }
      if (orbitDrag) {
        const dx = event.clientX - orbitDrag.x;
        const dy = event.clientY - orbitDrag.y;
        controls!.set({ yaw: orbitDrag.yaw - dx * 0.0035, pitch: orbitDrag.pitch + dy * 0.0035 });
        return;
      }
      if (pitchDrag && pitchSpec) {
        event.stopImmediatePropagation();
        const plane = rayPlaneY(rayAt(event), pitchDrag.height);
        if (!plane) return;
        // Fader position relative to its neutral (pitch 0) knob center.
        const neutral = pitchInstances[0] ? vec3.transformMat4(instanceCenterLocal(pitchInstances[0]), pitchInstances[0].base) : vec3.create();
        const along = vec3.dot(vec3.sub(plane, neutral), pitchAxis) - pitchDrag.grabOffset;
        const travel = pitchSpec.halfTravel * scale;
        setPitchImpl!((Math.max(-travel, Math.min(travel, along)) / travel) * pitchSpec.maxPitch);
        return;
      }
      if (pointerDown) return; // orbiting
      const hit = pick(rayAt(event), true);
      canvas.style.cursor = hit ? (hit.kind === "pitch" ? "ew-resize" : "pointer") : "";
    };
    const onPointerUp = (event: PointerEvent) => {
      if (table) {
        touches.delete(event.pointerId);
        try { if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
        if (pinch) { if (touches.size < 2) pinch = undefined; return; }
        if (orbitDrag) { orbitDrag = undefined; return; }
      }
      if (pitchDrag) {
        event.stopImmediatePropagation();
        pitchDrag = undefined;
        canvas.releasePointerCapture(event.pointerId);
        canvas.style.cursor = "";
        return;
      }
      const down = pointerDown;
      pointerDown = undefined;
      if (!down || !isTap(down, { x: event.clientX, y: event.clientY, time: performance.now() })) return;
      const hit = pick(rayAt(event), false);
      if (hit && new URLSearchParams(location.search).has("debug")) {
        const ray = rayAt(event);
        const world = vec3.addScaled(ray.origin, ray.direction, hit.t);
        const file = vec3.transformMat4(world, mat4.invert(root));
        const plateHit = rayPlaneY(ray, 0.089 * scale);
        const plate = plateHit ? Array.from(vec3.transformMat4(plateHit, mat4.invert(root))).map((v) => +v.toFixed(4)) : null;
        (window as unknown as { __lastPick?: unknown }).__lastPick = { kind: hit.kind, name: hit.inst?.name, mesh: hit.inst?.meshIndex, file: Array.from(file).map((v) => +v.toFixed(4)), plate };
      }
      if (hit?.kind === "tonearm") toggleArmImpl!();
      else if (hit?.kind === "button" && hit.button) pressButton(hit.button);
    };
    canvas.addEventListener("pointerdown", onPointerDown, { capture: true });
    canvas.addEventListener("pointermove", onPointerMove, { capture: true });
    canvas.addEventListener("pointerup", onPointerUp, { capture: true });
    canvas.addEventListener("pointercancel", onPointerUp, { capture: true });
    // Table theme: mouse wheel zooms (touch users pinch).
    const onWheel = (event: WheelEvent) => {
      if (!table) return;
      event.preventDefault();
      controls!.set({ distance: controls!.distance * Math.exp(event.deltaY * 0.0012) });
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    cleanups.push(() => canvas.removeEventListener("wheel", onWheel));
    cleanups.push(() => canvas.removeEventListener("pointercancel", onPointerUp, { capture: true }));
    cleanups.push(() => {
      canvas.removeEventListener("pointerdown", onPointerDown, { capture: true });
      canvas.removeEventListener("pointermove", onPointerMove, { capture: true });
      canvas.removeEventListener("pointerup", onPointerUp, { capture: true });
    });
    emitState();
    let fpsAccumulator = 0;
    let fpsFrames = 0;
    let fpsLast = performance.now();
    let qualitySettleUntil = performance.now() + 2500; // ignore warm-up frames
    let qualityCeiling = QUALITY_LADDER.length - 1;
    let qualityCeilingUntil = 0;
    const cameraDistance = vec3.create();
    const blended = drawables.filter((d) => d.blend);

    loop = frameLoop(ctx, (f) => {
      const dt = Math.min(time.deltaTime, 0.1);
      const now = performance.now();

      // Camera
      if (settings.autoRotate && !table && now - lastInteraction > 2500) {
        controls!.set({ yaw: controls!.yaw + dt * 0.18 });
      }
      controls!.update(dt);

      // Motor: ease toward the target speed (direct drive spins up fast, coasts down slower).
      if (settings.spin !== deck.running) { deck.running = settings.spin; emitState(); }
      const effectivePitch = deck.quartz ? 0 : deck.pitch;
      const targetOmega = deck.running ? (deck.speed / 60) * Math.PI * 2 * (1 + effectivePitch) : 0;
      // Press feedback: LEDs of a just-tapped button blink off for a moment.
      if (buttons.some((b) => now - b.pressedAt < PRESS_BLINK * 1000 + 50)) updateLeds(now);
      const rate = deck.running ? 6 : 1.4;
      deck.omega += (targetOmega - deck.omega) * Math.min(1, dt * rate);
      if (!deck.running && deck.omega < 0.01) deck.omega = 0;
      spinAngle += dt * deck.omega;
      ledPower += ((deck.running ? 1 : 0) - ledPower) * Math.min(1, dt * 8);

      // Audio follows the stylus: plays only with the arm down and settled, at platter speed.
      const nominalOmega = (PLATTER_RPM / 60) * Math.PI * 2;
      const stylusOnRecord = deck.armDown && !armTweens.busy;
      if (audioStateDirty) { audioStateDirty = false; emitState(); }
      // The stylus follows the groove at the platter's speed: real records take ~20 min per side.
      if (stylusOnRecord && deck.omega > 0) {
        const next = Math.min(1, progress + (dt * (deck.omega / nominalOmega)) / SIDE_SECONDS);
        if (Math.abs(next - progress) > 0.001) { progress = next; emitState(); } else progress = next;
      }
      if (audio?.enabled) {
        audio.setNeedleDown(stylusOnRecord);
        audio.setRate(deck.omega / nominalOmega);
        audio.update(dt);
      }

      // Tonearm choreography, plus the slow creep toward the run-out while playing.
      if (armTweens.busy) {
        armTweens.update(dt);
        applyArmPose();
        for (const inst of armInstances) dirtyStatic.add(inst);
        if (!armTweens.busy) emitState();
      } else if (stylusOnRecord) {
        const yaw = armYawAt(progress);
        if (Math.abs(yaw - deck.armValues.yaw) > 1e-5) {
          deck.armValues.yaw = yaw;
          applyArmPose();
          for (const inst of armInstances) dirtyStatic.add(inst);
        }
      }
      for (const inst of pitchInstances) dirtyStatic.add(inst);

      // Parts that only exist while the arm is parked (the rest clip) vanish once it leaves.
      const armAway = Math.abs(deck.armValues.yaw) > 0.03;
      for (const d of restOnlyDrawables) d.hidden = armAway;

      // Spinning parts every frame; static parts only when their pose changed.
      for (const inst of instances) {
        if (inst.spin) {
          mat4.translation(inst.pivot, spinMatrix);
          mat4.rotateY(spinMatrix, spinAngle, spinMatrix);
          mat4.translate(spinMatrix, [-inst.pivot[0], -inst.pivot[1], -inst.pivot[2]], spinMatrix);
          writeInstance(inst, spinMatrix);
        } else if (dirtyStatic.has(inst)) {
          writeInstance(inst, undefined);
        }
      }
      dirtyStatic.clear();

      // Key light follows the rotated studio rig.
      const lightDir = rotateY(keyLightDirection(layout.light), settings.envRotation);
      const eye = vec3.addScaled(focus, lightDir, radius * 2.5);
      mat4.lookAt(eye, focus, [0, 1, 0], lightView);
      mat4.multiply(lightProjection, lightView, lightViewProjection);
      lightUniforms.set({ viewProjection: new Float32Array(lightViewProjection) });

      const cameraPosition = camera.worldPosition;
      sceneUniforms.set({
        viewProjection: new Float32Array(camera.viewProjection),
        lightViewProjection: new Float32Array(lightViewProjection),
        cameraPosition: [cameraPosition[0], cameraPosition[1], cameraPosition[2]],
        envRotation: settings.envRotation,
        lightDirection: [lightDir[0], lightDir[1], lightDir[2]],
        lightIntensity: settings.lightIntensity,
        envIntensity: settings.envIntensity,
        finish: settings.finish === "black" ? 1 : 0,
        time: time.time,
        ledPower,
      });
      post.set({ params: postParams(sceneTarget.size[0] / sceneTarget.size[1]) });

      // Transparent surfaces back-to-front.
      if (blended.length > 1) {
        blended.sort((a, b) => vec3.lengthSq(vec3.sub(b.center, cameraPosition, cameraDistance)) - vec3.lengthSq(vec3.sub(a.center, cameraPosition, cameraDistance)));
      }

      f.pass({ target: shadowTarget, clear: [0, 0, 0, 0] }, (pass) => {
        for (const d of drawables) if (d.shadow && !d.hidden) pass.draw(d.shadow);
      });
      f.pass({ target: sceneTarget }, (pass) => {
        pass.draw(floorDraw);
        for (const d of drawables) if (!d.blend && !d.hidden) pass.draw(d.main);
        for (const d of blended) if (!d.hidden) pass.draw(d.main);
      });
      if (bloomEnabled) {
        f.pass(bloomBright, bright);
        f.pass(bloomA, blurH);
        f.pass(bloomB, blurV);
      }
      f.pass(canvasSurface, post);

      // FPS (reported twice a second) + adaptive quality, measured only while visible.
      fpsFrames++;
      fpsAccumulator += now - fpsLast;
      fpsLast = now;
      if (fpsAccumulator >= 500) {
        const fps = (fpsFrames * 1000) / fpsAccumulator;
        callbacks.onStats?.({ fps });
        fpsAccumulator = 0;
        fpsFrames = 0;
        if (document.visibilityState === "visible" && now > qualitySettleUntil) {
          if (fps < FPS_FLOOR && qualityLevel > 0) {
            qualityCeiling = qualityLevel - 1; // this level was too much: do not retry it for a while
            qualityCeilingUntil = now + 20000;
            applyQuality(qualityLevel - 1);
            qualitySettleUntil = now + 1500;
          } else if (fps > FPS_HEADROOM && qualityLevel < QUALITY_LADDER.length - 1) {
            if (now > qualityCeilingUntil) qualityCeiling = QUALITY_LADDER.length - 1;
            if (qualityLevel < qualityCeiling) {
              applyQuality(qualityLevel + 1);
              qualitySettleUntil = now + 2500;
            }
          }
        }
      }
    }, mobileQuality ? { fps: 60 } : undefined);

    callbacks.onReady({
      adapter: adapterName,
      triangles: triangleCount,
      materials: asset.materials.length + attachmentAssets.reduce((n, a) => n + a.materials.length, 0),
      textures: textureCache.size,
      drawCalls: drawables.length + 1,
      extensions: asset.extensionsUsed,
      spinningNodes,
      hasTonearm: !!arm,
    });
    callbacks.onProgress({ phase: "ready" });
    console.info(`[viewer] ${model.title}: ${triangleCount.toLocaleString()} triangles, ${drawables.length + 1} draws, ready in ${Math.round(performance.now() - t0)} ms`);
  };

  run().catch((error: unknown) => {
    if (disposed) return;
    console.error(error);
    callbacks.onError(error instanceof Error ? error : new Error(String(error)));
  });

  return {
    update(partial) {
      Object.assign(settings, partial);
    },
    resetCamera() {
      resetCameraImpl?.();
    },
    togglePlay() {
      togglePlayImpl?.();
    },
    toggleArm() {
      toggleArmImpl?.();
    },
    setPitch(pitch) {
      setPitchImpl?.(pitch);
    },
    async enableAudio() {
      if (!audio) return;
      await audio.enable();
      audioStateDirty = true;
    },
    spectrum(out) {
      return audio?.spectrum(out) ?? false;
    },
    dispose() {
      disposed = true;
      loop?.stop();
      controls?.dispose();
      for (const c of cleanups) c();
      audio?.dispose();
      gpu?.dispose();
    },
  };
}

// --- helpers ---------------------------------------------------------------------------------------------

async function fetchWithProgress(url: string, onProgress: (loaded: number, total?: number) => void): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`No se pudo descargar el modelo (${response.status}) desde ${url}`);
  const total = Number(response.headers.get("content-length")) || undefined;
  if (!response.body) return response.arrayBuffer();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out.buffer;
}

/** Yield to the event loop so React can paint progress. Uses a timer, not rAF: rAF never fires in hidden tabs. */
function nextFrame(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function collectTextureUsage(asset: GlbAsset): number {
  const keys = new Set<string>();
  for (const m of asset.materials) {
    const add = (i: number, srgb: boolean) => {
      if (i < 0) return;
      const ref = asset.textures[i];
      if (ref && ref.source >= 0) keys.add(`${ref.source}:${srgb}`);
    };
    add(m.baseColorTexture, true);
    add(m.emissiveTexture, true);
    add(m.metallicRoughnessTexture, false);
    add(m.normalTexture, false);
    add(m.occlusionTexture, false);
  }
  return keys.size;
}

function normalMatrixOf(m: Mat4): Float32Array {
  const inv = mat4.invert(m);
  return new Float32Array(mat4.transpose(inv));
}

function transformBounds(min: readonly number[], max: readonly number[], m: Mat4): Bounds {
  const outMin = [Infinity, Infinity, Infinity];
  const outMax = [-Infinity, -Infinity, -Infinity];
  const corner = vec3.create();
  for (let c = 0; c < 8; c++) {
    corner[0] = c & 1 ? max[0] : min[0];
    corner[1] = c & 2 ? max[1] : min[1];
    corner[2] = c & 4 ? max[2] : min[2];
    const w = vec3.transformMat4(corner, m);
    for (let k = 0; k < 3; k++) {
      if (w[k] < outMin[k]) outMin[k] = w[k];
      if (w[k] > outMax[k]) outMax[k] = w[k];
    }
  }
  return { min: outMin, max: outMax };
}

function instanceBounds(mesh: { primitives: { min: number[]; max: number[] }[] }, world: Mat4): Bounds {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const prim of mesh.primitives) {
    const b = transformBounds(prim.min, prim.max, world);
    for (let k = 0; k < 3; k++) {
      if (b.min[k] < min[k]) min[k] = b.min[k];
      if (b.max[k] > max[k]) max[k] = b.max[k];
    }
  }
  return { min, max };
}

function instanceCenterLocal(inst: { bounds: Bounds[] }): Float32Array {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const b of inst.bounds) {
    for (let k = 0; k < 3; k++) {
      if (b.min[k] < min[k]) min[k] = b.min[k];
      if (b.max[k] > max[k]) max[k] = b.max[k];
    }
  }
  return vec3.create((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
}

function instanceCenter(mesh: { primitives: { min: number[]; max: number[] }[] }, world: Mat4): Float32Array {
  const { min, max } = instanceBounds(mesh, world);
  return new Float32Array([(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]);
}

/**
 * World matrices for an attachment: re-oriented to Y up, optionally rescaled to a real
 * diameter, centered on the platter axis and resting on the platter's top surface.
 */
function placeAttachment(
  attachment: GlbAsset,
  spec: AttachmentSpec,
  ctx: { scale: number; platterPivot?: Float32Array; platterTop: number },
): { worlds: Mat4[]; top: number } | undefined {
  if (spec.placeOnPlatter && (!ctx.platterPivot || !Number.isFinite(ctx.platterTop))) return undefined;
  const orientation = spec.upAxis === "z" ? mat4.rotationX(-Math.PI / 2) : mat4.identity();
  const oriented = transformBounds(attachment.min, attachment.max, orientation);
  const sizeX = oriented.max[0] - oriented.min[0];
  const sizeZ = oriented.max[2] - oriented.min[2];
  let s = ctx.scale;
  if (spec.fitDiameter) s = (spec.fitDiameter * ctx.scale) / Math.max(sizeX, sizeZ);
  const cx = (oriented.min[0] + oriented.max[0]) / 2;
  const cz = (oriented.min[2] + oriented.max[2]) / 2;
  const lift = (spec.lift ?? 0) * ctx.scale;
  const position = spec.placeOnPlatter && ctx.platterPivot
    ? [ctx.platterPivot[0], ctx.platterTop + lift, ctx.platterPivot[2]]
    : [0, lift, 0];
  const m = mat4.translation(position);
  mat4.scale(m, [s, s, s], m);
  mat4.translate(m, [-cx, -oriented.min[1], -cz], m);
  mat4.multiply(m, orientation, m);
  return {
    worlds: attachment.instances.map((inst) => mat4.multiply(m, inst.worldMatrix)),
    top: position[1] + (oriented.max[1] - oriented.min[1]) * s,
  };
}

/** Rotation about +Y around `pivot` that puts `stylus` at `radius` from `center` (XZ plane). */
function solveTonearmAngle(pivot: Float32Array, stylus: Float32Array, center: Float32Array, radius: number): number {
  const dx = stylus[0] - pivot[0];
  const dz = stylus[2] - pivot[2];
  let best = 0;
  let bestError = Infinity;
  for (let i = -1500; i <= 1500; i++) {
    const theta = (i / 1500) * (Math.PI / 2);
    const c = Math.cos(theta), s = Math.sin(theta);
    const x = pivot[0] + dx * c + dz * s;
    const z = pivot[2] - dx * s + dz * c;
    const error = Math.abs(Math.hypot(x - center[0], z - center[2]) - radius);
    if (error < bestError - 1e-9 || (Math.abs(error - bestError) < 1e-9 && Math.abs(theta) < Math.abs(best))) {
      bestError = error;
      best = theta;
    }
  }
  return best;
}

/** Defaults match the key soft box in env.wgsl: azimuth -35°, elevation 42°. */
function keyLightDirection(light?: { azimuth: number; elevation: number }): Float32Array {
  const az = ((light?.azimuth ?? -35) * Math.PI) / 180;
  const el = ((light?.elevation ?? 42) * Math.PI) / 180;
  return vec3.normalize(vec3.create(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)));
}

function rotateY(v: Float32Array, angle: number): Float32Array {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return vec3.create(c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]);
}

/**
 * Splits triangles of `meshIndex` whose vertices (in scene space) fall inside the box into a
 * new mesh, instanced with the same transforms. Returns the new mesh index or -1.
 */
function splitMeshByRegion(asset: GlbAsset, meshIndex: number, min: readonly number[], max: readonly number[]): number {
  const mesh = asset.meshes[meshIndex];
  const owners = asset.instances.filter((i) => i.mesh === meshIndex);
  if (!mesh || owners.length === 0) return -1;
  const world = owners[0].worldMatrix;
  const insidePrims: typeof mesh.primitives = [];
  const outsidePrims: typeof mesh.primitives = [];
  const p = vec3.create();
  for (const prim of mesh.primitives) {
    const inside: number[] = [];
    const outside: number[] = [];
    const v = prim.vertices;
    const idx = prim.indices;
    const bmin = [Infinity, Infinity, Infinity];
    const bmax = [-Infinity, -Infinity, -Infinity];
    for (let t = 0; t < idx.length; t += 3) {
      let count = 0;
      for (let k = 0; k < 3; k++) {
        const i = idx[t + k] * VERTEX_STRIDE_FLOATS;
        vec3.set(v[i], v[i + 1], v[i + 2], p);
        const w = vec3.transformMat4(p, world);
        if (w[0] >= min[0] && w[0] <= max[0] && w[1] >= min[1] && w[1] <= max[1] && w[2] >= min[2] && w[2] <= max[2]) count++;
      }
      if (count >= 2) {
        for (let k = 0; k < 3; k++) {
          const vi = idx[t + k];
          inside.push(vi);
          const i = vi * VERTEX_STRIDE_FLOATS;
          for (let a = 0; a < 3; a++) {
            if (v[i + a] < bmin[a]) bmin[a] = v[i + a];
            if (v[i + a] > bmax[a]) bmax[a] = v[i + a];
          }
        }
      } else {
        outside.push(idx[t], idx[t + 1], idx[t + 2]);
      }
    }
    if (inside.length) insidePrims.push({ ...prim, indices: new Uint32Array(inside), min: bmin as [number, number, number], max: bmax as [number, number, number] });
    if (outside.length) outsidePrims.push({ ...prim, indices: new Uint32Array(outside) });
  }
  if (insidePrims.length === 0) return -1;
  mesh.primitives = outsidePrims;
  asset.meshes.push({ name: `${mesh.name}:arm`, primitives: insidePrims });
  const created = asset.meshes.length - 1;
  for (const inst of owners) asset.instances.push({ ...inst, mesh: created, name: `${inst.name}:arm` });
  return created;
}

/** Flat ring in the XZ plane with a square, center-cropped UV mapping (for cover art). */
function annulusVertices(inner: number, outer: number, segments: number, aspect = 1): Float32Array {
  const out = new Float32Array((segments + 1) * 2 * VERTEX_STRIDE_FLOATS);
  let o = 0;
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    for (const r of [inner, outer]) {
      const x = c * r, z = s * r;
      // Center-crop a landscape image (aspect = height / width) to the square label.
      const u = 0.5 + (x / outer) * 0.5 * Math.min(1, aspect);
      const v = 0.5 + (z / outer) * 0.5 * Math.min(1, 1 / aspect);
      out.set([x, 0, z, 0, 1, 0, u, v, 1, 0, 0, 1], o);
      o += VERTEX_STRIDE_FLOATS;
    }
  }
  return out;
}

function annulusIndices(segments: number): Uint32Array {
  const out = new Uint32Array(segments * 6);
  for (let i = 0; i < segments; i++) {
    const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
    // Counter-clockwise seen from +Y.
    out.set([a, c, b, b, c, d], i * 6);
  }
  return out;
}

/** Quad in the XZ plane, normal +Y, tangent +X, matching the shared vertex layout. */
function floorVertices(halfSize: number): Float32Array {
  const s = halfSize;
  // position(3) normal(3) uv(2) tangent(4)
  return new Float32Array([
    -s, 0, -s,  0, 1, 0,  0, 0,  1, 0, 0, 1,
    -s, 0,  s,  0, 1, 0,  0, 1,  1, 0, 0, 1,
     s, 0,  s,  0, 1, 0,  1, 1,  1, 0, 0, 1,
     s, 0, -s,  0, 1, 0,  1, 0,  1, 0, 0, 1,
  ]);
}
