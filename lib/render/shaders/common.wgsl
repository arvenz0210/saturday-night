// Pure WGSL module: shared math for the PBR pipeline (no bindings allowed here).

export fn pi() -> f32 { return 3.14159265359; }

export fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Van der Corput radical inverse -> Hammersley point set.
export fn hammersley(i: u32, n: u32) -> vec2f {
  var bits = i;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  let radical = f32(bits) * 2.3283064365386963e-10;
  return vec2f(f32(i) / f32(n), radical);
}

// Tangent frame around n, used to lift hemisphere samples into world space.
export fn tangentFrame(n: vec3f) -> mat3x3f {
  var up = vec3f(0.0, 0.0, 1.0);
  if (abs(n.z) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  let t = normalize(cross(up, n));
  let b = cross(n, t);
  return mat3x3f(t, b, n);
}

// GGX importance sample: returns a half vector around n for the given roughness.
export fn importanceSampleGGX(xi: vec2f, n: vec3f, roughness: f32) -> vec3f {
  let a = roughness * roughness;
  let phi = 2.0 * pi() * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let h = vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
  return normalize(tangentFrame(n) * h);
}

// Cosine-weighted hemisphere sample around n.
export fn cosineSampleHemisphere(xi: vec2f, n: vec3f) -> vec3f {
  let phi = 2.0 * pi() * xi.x;
  let cosTheta = sqrt(1.0 - xi.y);
  let sinTheta = sqrt(xi.y);
  let l = vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
  return normalize(tangentFrame(n) * l);
}

// Equirectangular mapping, Y up. u wraps around the horizon, v = 0 at the zenith.
export fn dirToEquirectUv(d: vec3f) -> vec2f {
  let u = atan2(d.x, -d.z) / (2.0 * pi()) + 0.5;
  let v = acos(clamp(d.y, -1.0, 1.0)) / pi();
  return vec2f(u, v);
}

export fn equirectUvToDir(uv: vec2f) -> vec3f {
  let phi = (uv.x - 0.5) * 2.0 * pi();
  let theta = uv.y * pi();
  let st = sin(theta);
  return vec3f(st * sin(phi), cos(theta), -st * cos(phi));
}

export fn rotateY(v: vec3f, angle: f32) -> vec3f {
  let c = cos(angle);
  let s = sin(angle);
  return vec3f(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

// --- Microfacet BRDF terms ---------------------------------------------------

export fn distributionGGX(nDotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(pi() * d * d, 1e-6);
}

// Height-correlated Smith visibility (already includes the 1 / (4 NdotL NdotV)).
export fn visibilitySmithGGXCorrelated(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let ggxV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
  let ggxL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
  return 0.5 / max(ggxV + ggxL, 1e-6);
}

// Schlick-GGX geometry term used for the split-sum BRDF LUT (k = a^2 / 2 for IBL).
export fn geometrySchlickGGXIbl(nDotV: f32, roughness: f32) -> f32 {
  let a = roughness;
  let k = (a * a) / 2.0;
  return nDotV / (nDotV * (1.0 - k) + k);
}

export fn fresnelSchlick(f0: vec3f, vDotH: f32) -> vec3f {
  let f = pow(1.0 - vDotH, 5.0);
  return f0 + (1.0 - f0) * f;
}

export fn fresnelSchlickRoughness(f0: vec3f, nDotV: f32, roughness: f32) -> vec3f {
  let fr = max(vec3f(1.0 - roughness), f0) - f0;
  return f0 + fr * pow(1.0 - nDotV, 5.0);
}

// --- Color ----------------------------------------------------------------------

export fn acesFitted(x: vec3f) -> vec3f {
  // Stephen Hill's ACES fit (sRGB working space).
  let m1 = mat3x3f(
    vec3f(0.59719, 0.07600, 0.02840),
    vec3f(0.35458, 0.90834, 0.13383),
    vec3f(0.04823, 0.01566, 0.83777)
  );
  let m2 = mat3x3f(
    vec3f(1.60475, -0.10208, -0.00327),
    vec3f(-0.53108, 1.10813, -0.07276),
    vec3f(-0.07367, -0.00605, 1.07602)
  );
  let v = m1 * x;
  let a = v * (v + 0.0245786) - 0.000090537;
  let b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(m2 * (a / b), vec3f(0.0), vec3f(1.0));
}

export fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

export fn rgbToHsv(c: vec3f) -> vec3f {
  let k = vec4f(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p = select(vec4f(c.gb, k.xy), vec4f(c.bg, k.wz), c.b > c.g);
  let q = select(vec4f(c.r, p.yzx), vec4f(p.xyw, c.r), p.x > c.r);
  let d = q.x - min(q.w, q.y);
  let e = 1.0e-10;
  return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// 0..1 mask for "gold / brass" tones: warm hue with real saturation.
export fn goldMask(c: vec3f) -> f32 {
  let hsv = rgbToHsv(c);
  let hue = hsv.x * 360.0;
  let hueMask = smoothstep(18.0, 30.0, hue) * (1.0 - smoothstep(58.0, 72.0, hue));
  let satMask = smoothstep(0.22, 0.42, hsv.y);
  return hueMask * satMask;
}

// Interleaved gradient noise (Jimenez) for per-pixel rotation of shadow taps.
export fn interleavedGradientNoise(p: vec2f) -> f32 {
  let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
  return fract(magic.z * fract(dot(p, magic.xy)));
}
