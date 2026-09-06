// Bloom step 2: separable 9-tap Gaussian, run once horizontally and once vertically.
struct Params {
  direction: vec2f, // texel-sized step, e.g. (texelX, 0) or (0, texelY)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let d = params.direction;
  var c = textureSampleLevel(src, samp, uv, 0.0).rgb * 0.2270270270;
  c += textureSampleLevel(src, samp, uv + d * 1.3846153846, 0.0).rgb * 0.3162162162;
  c += textureSampleLevel(src, samp, uv - d * 1.3846153846, 0.0).rgb * 0.3162162162;
  c += textureSampleLevel(src, samp, uv + d * 3.2307692308, 0.0).rgb * 0.0702702703;
  c += textureSampleLevel(src, samp, uv - d * 3.2307692308, 0.0).rgb * 0.0702702703;
  return vec4f(c, 1.0);
}
