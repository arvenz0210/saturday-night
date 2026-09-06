// Metallic-roughness PBR with image-based lighting, a shadowed key light and
// normal mapping. Outputs linear HDR into an rgba16float target.
import {
  dirToEquirectUv, distributionGGX, fresnelSchlick, fresnelSchlickRoughness,
  goldMask, interleavedGradientNoise, luminance, pi, rotateY, visibilitySmithGGXCorrelated
} from "./common.wgsl";

struct Scene {
  viewProjection: mat4x4f,
  lightViewProjection: mat4x4f,
  cameraPosition: vec3f,
  envRotation: f32,
  lightDirection: vec3f,
  lightIntensity: f32,
  lightColor: vec3f,
  envIntensity: f32,
  shadowTexel: vec2f,
  finish: f32,
  specularMips: f32,
  backgroundColor: vec3f,
  time: f32,
  /** 0 = studio floor (PBR), 1 = white table: floor writes shadow visibility with alpha 0. */
  floorMode: f32,
  /** PCF kernel radius in shadow-map texels. */
  shadowSoftness: f32,
  /** 0..1 brightness of LED materials (motor off = dark plastic). */
  ledPower: f32,
  /** World-space boxes whose fragments are discarded while hideBoxActive > 0.5. */
  hideBoxMin: vec3f,
  hideBoxActive: f32,
  hideBoxMax: vec3f,
  hideBox2Min: vec3f,
  hideBox2Max: vec3f,
}

struct Material {
  baseColorFactor: vec4f,
  emissiveFactor: vec3f,
  metallicFactor: f32,
  roughnessFactor: f32,
  normalScale: f32,
  occlusionStrength: f32,
  alphaCutoff: f32,
  flags: u32,
}

struct Model {
  model: mat4x4f,
  normalMatrix: mat4x4f,
}

// Material.flags bits
const HAS_BASE_COLOR_TEX: u32 = 1u;
const HAS_METALLIC_ROUGHNESS_TEX: u32 = 2u;
const HAS_NORMAL_TEX: u32 = 4u;
const HAS_EMISSIVE_TEX: u32 = 8u;
const HAS_OCCLUSION_TEX: u32 = 16u;
const ALPHA_MASK: u32 = 32u;
const IS_FLOOR: u32 = 64u;
const HAS_TANGENTS: u32 = 128u;
const RECOLORABLE: u32 = 256u;
const IS_LED: u32 = 512u;

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;
@group(0) @binding(3) var envSpecular: texture_2d<f32>;
@group(0) @binding(4) var envIrradiance: texture_2d<f32>;
@group(0) @binding(5) var envSampler: sampler;
@group(0) @binding(6) var brdfLut: texture_2d<f32>;
@group(0) @binding(7) var lutSampler: sampler;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var baseColorTex: texture_2d<f32>;
@group(1) @binding(2) var metallicRoughnessTex: texture_2d<f32>;
@group(1) @binding(3) var normalTex: texture_2d<f32>;
@group(1) @binding(4) var emissiveTex: texture_2d<f32>;
@group(1) @binding(5) var occlusionTex: texture_2d<f32>;
@group(1) @binding(6) var materialSampler: sampler;

@group(2) @binding(0) var<uniform> model: Model;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) worldNormal: vec3f,
  @location(2) uv: vec2f,
  @location(3) worldTangent: vec4f,
}

@vertex fn vs_main(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) tangent: vec4f,
) -> VertexOut {
  var out: VertexOut;
  let world = model.model * vec4f(position, 1.0);
  out.position = scene.viewProjection * world;
  out.worldPosition = world.xyz;
  out.worldNormal = normalize((model.normalMatrix * vec4f(normal, 0.0)).xyz);
  out.worldTangent = vec4f(normalize((model.model * vec4f(tangent.xyz, 0.0)).xyz), tangent.w);
  out.uv = uv;
  return out;
}

const POISSON: array<vec2f, 16> = array<vec2f, 16>(
  vec2f(-0.94201624, -0.39906216), vec2f(0.94558609, -0.76890725),
  vec2f(-0.09418410, -0.92938870), vec2f(0.34495938, 0.29387760),
  vec2f(-0.91588581, 0.45771432), vec2f(-0.81544232, -0.87912464),
  vec2f(-0.38277543, 0.27676845), vec2f(0.97484398, 0.75648379),
  vec2f(0.44323325, -0.97511554), vec2f(0.53742981, -0.47373420),
  vec2f(-0.26496911, -0.41893023), vec2f(0.79197514, 0.19090188),
  vec2f(-0.24188840, 0.99706507), vec2f(-0.81409955, 0.91437590),
  vec2f(0.19984126, 0.78641367), vec2f(0.14383161, -0.14100790)
);

// Percentage-closer soft shadow: 16 rotated Poisson taps, normal-offset to avoid acne.
fn shadowFactor(worldPosition: vec3f, n: vec3f, nDotL: f32, fragCoord: vec2f) -> f32 {
  let offsetScale = 0.006 * (1.0 - nDotL) + 0.0015;
  let lightSpace = scene.lightViewProjection * vec4f(worldPosition + n * offsetScale, 1.0);
  let p = lightSpace.xyz / lightSpace.w;
  let uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || p.z > 1.0 || p.z < 0.0) {
    return 1.0;
  }
  let angle = interleavedGradientNoise(fragCoord) * 2.0 * pi();
  let rot = mat2x2f(vec2f(cos(angle), sin(angle)), vec2f(-sin(angle), cos(angle)));
  let radius = scene.shadowTexel * scene.shadowSoftness;
  var lit = 0.0;
  for (var i = 0u; i < 16u; i++) {
    let offset = (rot * POISSON[i]) * radius;
    lit += textureSampleCompareLevel(shadowMap, shadowSampler, uv + offset, p.z - 0.0008);
  }
  return lit / 16.0;
}

fn sampleEnvSpecular(dir: vec3f, roughness: f32) -> vec3f {
  let uv = dirToEquirectUv(rotateY(dir, -scene.envRotation));
  return textureSampleLevel(envSpecular, envSampler, uv, roughness * (scene.specularMips - 1.0)).rgb;
}

fn sampleEnvIrradiance(dir: vec3f) -> vec3f {
  let uv = dirToEquirectUv(rotateY(dir, -scene.envRotation));
  return textureSampleLevel(envIrradiance, envSampler, uv, 0.0).rgb;
}

@fragment fn fs_main(in: VertexOut, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4f {
  let flags = material.flags;
  if (scene.hideBoxActive > 0.5 && (flags & IS_FLOOR) == 0u) {
    let p = in.worldPosition;
    if ((all(p >= scene.hideBoxMin) && all(p <= scene.hideBoxMax)) || (all(p >= scene.hideBox2Min) && all(p <= scene.hideBox2Max))) {
      discard;
    }
  }

  // --- Base color / alpha -----------------------------------------------------
  var baseColor = material.baseColorFactor;
  if ((flags & HAS_BASE_COLOR_TEX) != 0u) {
    baseColor *= textureSample(baseColorTex, materialSampler, in.uv);
  }
  if ((flags & ALPHA_MASK) != 0u && baseColor.a < material.alphaCutoff) {
    discard;
  }

  // White-table mode: the floor is a pure shadow catcher, composited in post.
  if ((flags & IS_FLOOR) != 0u && scene.floorMode > 0.5) {
    let up = vec3f(0.0, 1.0, 0.0);
    let lit = shadowFactor(in.worldPosition, up, max(dot(up, normalize(scene.lightDirection)), 0.0), in.position.xy);
    // Contact darkening: distance-based falloff keeps the far floor clean.
    let fade = 1.0 - smoothstep(0.6, 2.2, length(in.worldPosition.xz));
    let visibility = mix(1.0, lit, fade);
    return vec4f(vec3f(visibility), 0.0);
  }

  var metallic = material.metallicFactor;
  var roughness = material.roughnessFactor;
  if ((flags & HAS_METALLIC_ROUGHNESS_TEX) != 0u) {
    let mr = textureSample(metallicRoughnessTex, materialSampler, in.uv);
    roughness *= mr.g;
    metallic *= mr.b;
  }

  // "Negro" finish: key the gold/brass plating and turn it into satin black,
  // leaving chrome, rubber and labels untouched.
  if (scene.finish > 0.5 && (flags & RECOLORABLE) != 0u) {
    let mask = goldMask(baseColor.rgb);
    let lum = luminance(baseColor.rgb);
    let black = vec3f(0.018, 0.018, 0.02) + vec3f(0.035) * lum;
    baseColor = vec4f(mix(baseColor.rgb, black, mask), baseColor.a);
    metallic = mix(metallic, 0.0, mask);
    roughness = mix(roughness, clamp(roughness * 0.75 + 0.28, 0.3, 0.75), mask);
  }

  // --- Floor: dark satin studio surface fading into the backdrop -------------
  var floorFade = 0.0;
  if ((flags & IS_FLOOR) != 0u) {
    let r = length(in.worldPosition.xz);
    let rings = 0.5 + 0.5 * sin(r * 40.0);
    baseColor = vec4f(vec3f(0.028, 0.028, 0.03) + vec3f(0.004) * rings, 1.0);
    metallic = 0.0;
    roughness = 0.5 + 0.15 * smoothstep(0.2, 1.8, r);
    floorFade = smoothstep(0.9, 3.0, r);
  }

  roughness = clamp(roughness, 0.03, 1.0);
  metallic = clamp(metallic, 0.0, 1.0);

  // --- Normal -------------------------------------------------------------------
  var n = normalize(in.worldNormal);
  if (!frontFacing) { n = -n; }
  if ((flags & HAS_NORMAL_TEX) != 0u && (flags & HAS_TANGENTS) != 0u) {
    var t = normalize(in.worldTangent.xyz);
    t = normalize(t - n * dot(n, t));
    let b = cross(n, t) * in.worldTangent.w;
    var tn = textureSample(normalTex, materialSampler, in.uv).xyz * 2.0 - 1.0;
    tn = vec3f(tn.xy * material.normalScale, tn.z);
    n = normalize(mat3x3f(t, b, n) * tn);
  }

  var occlusion = 1.0;
  if ((flags & HAS_OCCLUSION_TEX) != 0u) {
    let ao = textureSample(occlusionTex, materialSampler, in.uv).r;
    occlusion = 1.0 + material.occlusionStrength * (ao - 1.0);
  }

  var emissive = material.emissiveFactor;
  if ((flags & HAS_EMISSIVE_TEX) != 0u) {
    emissive *= textureSample(emissiveTex, materialSampler, in.uv).rgb;
  }
  if ((flags & IS_LED) != 0u) {
    // Unpowered LED: dark smoked plastic instead of the lit color.
    emissive *= scene.ledPower;
    baseColor = vec4f(mix(vec3f(0.03, 0.03, 0.035), baseColor.rgb, scene.ledPower), baseColor.a);
    roughness = mix(0.35, roughness, scene.ledPower);
  }

  // --- Lighting -----------------------------------------------------------------
  let v = normalize(scene.cameraPosition - in.worldPosition);
  let nDotV = max(dot(n, v), 1e-4);
  let albedo = baseColor.rgb;
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let diffuseColor = albedo * (1.0 - metallic);

  // Key light (directional, shadowed).
  let l = normalize(scene.lightDirection);
  let h = normalize(l + v);
  let nDotL = max(dot(n, l), 0.0);
  let nDotH = max(dot(n, h), 0.0);
  let vDotH = max(dot(v, h), 0.0);
  var direct = vec3f(0.0);
  if (nDotL > 0.0) {
    let shadow = shadowFactor(in.worldPosition, n, nDotL, in.position.xy);
    let d = distributionGGX(nDotH, roughness);
    let vis = visibilitySmithGGXCorrelated(nDotV, nDotL, roughness);
    let f = fresnelSchlick(f0, vDotH);
    let specular = d * vis * f;
    let diffuse = diffuseColor / pi();
    direct = (diffuse * (1.0 - f) + specular) * scene.lightColor * scene.lightIntensity * nDotL * shadow;
  }

  // Image-based lighting (split sum).
  let fIbl = fresnelSchlickRoughness(f0, nDotV, roughness);
  let kD = (1.0 - fIbl) * (1.0 - metallic);
  let irradiance = sampleEnvIrradiance(n);
  let ambientDiffuse = kD * albedo * irradiance;
  let r = reflect(-v, n);
  let prefiltered = sampleEnvSpecular(r, roughness);
  let brdf = textureSampleLevel(brdfLut, lutSampler, vec2f(nDotV, 1.0 - roughness), 0.0).rg;
  let ambientSpecular = prefiltered * (f0 * brdf.x + brdf.y);
  // Horizon fade: kill reflections that would come from below the surface.
  let horizon = clamp(1.0 + dot(r, in.worldNormal), 0.0, 1.0);
  var specularScale = horizon * horizon;
  if ((flags & IS_FLOOR) != 0u) { specularScale *= 0.45; } // keep the floor moody, not mirror-like
  let ambient = (ambientDiffuse + ambientSpecular * specularScale) * occlusion * scene.envIntensity;

  var color = direct + ambient + emissive;
  if ((flags & IS_FLOOR) != 0u) {
    color = mix(color, scene.backgroundColor, floorFade);
  }
  return vec4f(color, baseColor.a);
}
