// Bloom step 1: threshold + 13-tap downsample of the HDR scene into a quarter-res target.
import { luminance } from "./common.wgsl";

struct Params {
  texel: vec2f,
  threshold: f32,
  knee: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

fn tap(uv: vec2f) -> vec3f {
  // Alpha 0 = background / shadow catcher: never blooms.
  let s = textureSampleLevel(src, samp, uv, 0.0);
  let c = s.rgb * s.a;
  // Soft knee threshold (Unity-style).
  let l = luminance(c);
  let soft = clamp(l - params.threshold + params.knee, 0.0, 2.0 * params.knee);
  let contribution = max(soft * soft / (4.0 * params.knee + 1e-4), l - params.threshold) / max(l, 1e-4);
  // Fireflies are tamed with a luminance weight.
  return c * contribution / (1.0 + l);
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let t = params.texel;
  var c = tap(uv) * 0.125;
  c += (tap(uv + t * vec2f(-1.0, -1.0)) + tap(uv + t * vec2f(1.0, -1.0)) +
        tap(uv + t * vec2f(-1.0, 1.0)) + tap(uv + t * vec2f(1.0, 1.0))) * 0.125;
  c += (tap(uv + t * vec2f(-2.0, 0.0)) + tap(uv + t * vec2f(2.0, 0.0)) +
        tap(uv + t * vec2f(0.0, -2.0)) + tap(uv + t * vec2f(0.0, 2.0))) * 0.0625;
  c += (tap(uv + t * vec2f(-2.0, -2.0)) + tap(uv + t * vec2f(2.0, -2.0)) +
        tap(uv + t * vec2f(-2.0, 2.0)) + tap(uv + t * vec2f(2.0, 2.0))) * 0.03125;
  return vec4f(c, 1.0);
}
