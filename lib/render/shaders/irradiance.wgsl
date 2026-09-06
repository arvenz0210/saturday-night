// Split-sum IBL, part 2: cosine-convolved irradiance (divided by pi, so
// diffuse = albedo * sample). Cosine-weighted sampling keeps the estimator to a plain mean.
import { cosineSampleHemisphere, dirToEquirectUv, equirectUvToDir, hammersley } from "./common.wgsl";

struct Params {
  sampleCount: u32,
  lod: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var env: texture_2d<f32>;
@group(0) @binding(2) var envSampler: sampler;

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let n = equirectUvToDir(uv);
  var sum = vec3f(0.0);
  for (var i = 0u; i < params.sampleCount; i++) {
    let l = cosineSampleHemisphere(hammersley(i, params.sampleCount), n);
    sum += textureSampleLevel(env, envSampler, dirToEquirectUv(l), params.lod).rgb;
  }
  return vec4f(sum / f32(params.sampleCount), 1.0);
}
