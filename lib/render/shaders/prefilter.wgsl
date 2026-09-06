// Split-sum IBL, part 1: GGX-prefiltered specular radiance for one roughness level.
// Uses filtered importance sampling (mip level chosen from the sample pdf) so
// even 64 samples give a firefly-free result.
import { dirToEquirectUv, equirectUvToDir, hammersley, importanceSampleGGX, distributionGGX, pi } from "./common.wgsl";

struct Params {
  roughness: f32,
  sampleCount: u32,
  srcWidth: f32,
  srcHeight: f32,
  srcMips: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var env: texture_2d<f32>;
@group(0) @binding(2) var envSampler: sampler;

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let n = equirectUvToDir(uv);
  if (params.roughness < 0.001) {
    return vec4f(textureSampleLevel(env, envSampler, uv, 0.0).rgb, 1.0);
  }
  let v = n;
  var sum = vec3f(0.0);
  var weight = 0.0;
  // Average solid angle of one equirect texel (good enough for the lod heuristic).
  let saTexel = 4.0 * pi() / (params.srcWidth * params.srcHeight);
  for (var i = 0u; i < params.sampleCount; i++) {
    let xi = hammersley(i, params.sampleCount);
    let h = importanceSampleGGX(xi, n, params.roughness);
    let l = normalize(2.0 * dot(v, h) * h - v);
    let nDotL = dot(n, l);
    if (nDotL > 0.0) {
      let nDotH = max(dot(n, h), 0.0);
      let d = distributionGGX(nDotH, params.roughness);
      let pdf = d * nDotH / (4.0 * nDotH) + 0.0001;
      let saSample = 1.0 / (f32(params.sampleCount) * pdf + 0.0001);
      let lod = clamp(0.5 * log2(saSample / saTexel) + 1.0, 0.0, params.srcMips - 1.0);
      sum += textureSampleLevel(env, envSampler, dirToEquirectUv(l), lod).rgb * nDotL;
      weight += nDotL;
    }
  }
  return vec4f(sum / max(weight, 1e-4), 1.0);
}
