// Split-sum IBL, part 3: the environment BRDF lookup table.
// x = N.V, y = roughness  ->  (scale, bias) applied to F0.
import { geometrySchlickGGXIbl, hammersley, importanceSampleGGX } from "./common.wgsl";

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let nDotV = max(uv.x, 1e-3);
  let roughness = 1.0 - uv.y; // v grows downward: keep roughness 0 at the top row
  let v = vec3f(sqrt(1.0 - nDotV * nDotV), 0.0, nDotV);
  let n = vec3f(0.0, 0.0, 1.0);
  var a = 0.0;
  var b = 0.0;
  let count = 512u;
  for (var i = 0u; i < count; i++) {
    let xi = hammersley(i, count);
    let h = importanceSampleGGX(xi, n, roughness);
    let l = normalize(2.0 * dot(v, h) * h - v);
    let nDotL = max(l.z, 0.0);
    let nDotH = max(h.z, 0.0);
    let vDotH = max(dot(v, h), 0.0);
    if (nDotL > 0.0) {
      let g = geometrySchlickGGXIbl(nDotV, roughness) * geometrySchlickGGXIbl(nDotL, roughness);
      let gVis = (g * vDotH) / max(nDotH * nDotV, 1e-5);
      let fc = pow(1.0 - vDotH, 5.0);
      a += (1.0 - fc) * gVis;
      b += fc * gVis;
    }
  }
  return vec4f(a / f32(count), b / f32(count), 0.0, 1.0);
}
