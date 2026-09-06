// Final composite: exposure, bloom, ACES tone mapping, vignette, sRGB encode.
import { acesFitted, linearToSrgb } from "./common.wgsl";

struct Params {
  exposure: f32,
  bloomStrength: f32,
  vignette: f32,
  aspect: f32,
  /** sRGB background shown where the scene left alpha 0 (white-table mode). */
  background: vec3f,
  /** How dark the caught shadow gets on that background. */
  shadowStrength: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var bloom: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let sample = textureSampleLevel(scene, samp, uv, 0.0);
  var hdr = sample.rgb;
  hdr += textureSampleLevel(bloom, samp, uv, 0.0).rgb * params.bloomStrength;
  hdr *= params.exposure;

  // Gentle vignette in linear space so it stays smooth after tone mapping.
  let centered = (uv - 0.5) * vec2f(params.aspect, 1.0);
  let v = 1.0 - params.vignette * smoothstep(0.35, 1.15, length(centered));
  hdr *= v;

  let mapped = linearToSrgb(acesFitted(hdr));
  // Alpha 0 pixels carry shadow visibility in .r (floor catcher) or 1 (clear color).
  let catcher = params.background * (1.0 - params.shadowStrength * (1.0 - clamp(sample.r, 0.0, 1.0)));
  return vec4f(mix(catcher, mapped, clamp(sample.a, 0.0, 1.0)), 1.0);
}
