// Procedural studio environment rendered once into an HDR equirectangular map.
// Three soft boxes (key, fill, rim strip) over a dark cyclorama: the classic
// product-photography setup, which is what gives brushed metal its long highlights.
import { equirectUvToDir, pi } from "./common.wgsl";

fn dirFromAngles(azimuthDeg: f32, elevationDeg: f32) -> vec3f {
  let az = radians(azimuthDeg);
  let el = radians(elevationDeg);
  return normalize(vec3f(sin(az) * cos(el), sin(el), -cos(az) * cos(el)));
}

// Rectangular soft box: a smooth-edged rectangle of angular size (w, h) radians
// centered on `center`, expressed in that light's own tangent frame.
fn softBox(d: vec3f, center: vec3f, w: f32, h: f32, softness: f32) -> f32 {
  let cosAngle = dot(d, center);
  if (cosAngle <= 0.0) { return 0.0; }
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(center.y) > 0.99) { up = vec3f(0.0, 0.0, 1.0); }
  let right = normalize(cross(up, center));
  let realUp = cross(center, right);
  // Project the direction onto the light's plane at unit distance.
  let p = d / cosAngle;
  let x = dot(p, right);
  let y = dot(p, realUp);
  let fx = 1.0 - smoothstep(w - softness, w + softness, abs(x));
  let fy = 1.0 - smoothstep(h - softness, h + softness, abs(y));
  // Falloff toward the box edges gives the panel a gentle gradient.
  let vignette = 1.0 - 0.35 * clamp((abs(x) / w) * (abs(x) / w) + (abs(y) / h) * (abs(y) / h), 0.0, 1.0);
  return fx * fy * vignette;
}

@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let d = equirectUvToDir(uv);

  // Cyclorama: cool dark gradient above the horizon, warm dark floor below.
  let upness = clamp(d.y, -1.0, 1.0);
  let sky = mix(vec3f(0.035, 0.037, 0.045), vec3f(0.11, 0.12, 0.15), smoothstep(-0.05, 0.9, upness));
  let floor = mix(vec3f(0.028, 0.026, 0.024), vec3f(0.05, 0.047, 0.044), smoothstep(-1.0, -0.1, upness));
  var color = select(floor, sky, upness > 0.0);
  // Soft horizon glow (bounce off the backdrop).
  color += vec3f(0.06, 0.06, 0.07) * exp(-abs(upness) * 6.0);

  // Key light: large warm soft box, front-left, high.
  let keyDir = dirFromAngles(-35.0, 42.0);
  color += vec3f(1.0, 0.96, 0.9) * 9.0 * softBox(d, keyDir, 0.62, 0.48, 0.06);

  // Fill: cooler, dimmer panel from the right.
  let fillDir = dirFromAngles(62.0, 18.0);
  color += vec3f(0.85, 0.9, 1.0) * 2.2 * softBox(d, fillDir, 0.75, 0.55, 0.1);

  // Rim: long thin strip behind and above, produces the specular streaks.
  let rimDir = dirFromAngles(178.0, 58.0);
  color += vec3f(1.0, 0.98, 0.95) * 6.5 * softBox(d, rimDir, 1.5, 0.12, 0.03);

  // Small bright top kicker for crisp sparkle on chrome.
  let topDir = dirFromAngles(110.0, 78.0);
  color += vec3f(1.0) * 4.0 * softBox(d, topDir, 0.18, 0.18, 0.04);

  return vec4f(color, 1.0);
}
