// Picking and gesture helpers: screen rays, ray/box and ray/plane tests, tap detection.
import { mat4, vec3, type Mat4 } from "wgpu-matrix";

export interface Ray {
  origin: Float32Array;
  direction: Float32Array;
}

/** World-space ray through a canvas pixel (CSS coordinates relative to the canvas). */
export function screenRay(x: number, y: number, width: number, height: number, viewProjection: Mat4, cameraPosition: Float32Array): Ray {
  const ndcX = (x / width) * 2 - 1;
  const ndcY = 1 - (y / height) * 2;
  const inverse = mat4.invert(viewProjection);
  const far = unproject(ndcX, ndcY, 1, inverse);
  const direction = vec3.normalize(vec3.sub(far, cameraPosition));
  return { origin: vec3.copy(cameraPosition), direction };
}

function unproject(x: number, y: number, z: number, inverse: Mat4): Float32Array {
  const px = inverse[0] * x + inverse[4] * y + inverse[8] * z + inverse[12];
  const py = inverse[1] * x + inverse[5] * y + inverse[9] * z + inverse[13];
  const pz = inverse[2] * x + inverse[6] * y + inverse[10] * z + inverse[14];
  const pw = inverse[3] * x + inverse[7] * y + inverse[11] * z + inverse[15];
  return vec3.create(px / pw, py / pw, pz / pw);
}

/** Slab test in the box's local space; returns the entry distance along the ray or null. */
export function rayLocalBox(ray: Ray, worldInverse: Mat4, min: readonly number[], max: readonly number[], pad = 0): number | null {
  const o = vec3.transformMat4(ray.origin, worldInverse);
  // Direction transforms without translation.
  const d = vec3.create(
    worldInverse[0] * ray.direction[0] + worldInverse[4] * ray.direction[1] + worldInverse[8] * ray.direction[2],
    worldInverse[1] * ray.direction[0] + worldInverse[5] * ray.direction[1] + worldInverse[9] * ray.direction[2],
    worldInverse[2] * ray.direction[0] + worldInverse[6] * ray.direction[1] + worldInverse[10] * ray.direction[2],
  );
  let tMin = -Infinity;
  let tMax = Infinity;
  for (let k = 0; k < 3; k++) {
    const lo = min[k] - pad;
    const hi = max[k] + pad;
    if (Math.abs(d[k]) < 1e-12) {
      if (o[k] < lo || o[k] > hi) return null;
      continue;
    }
    let t1 = (lo - o[k]) / d[k];
    let t2 = (hi - o[k]) / d[k];
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return null;
  }
  if (tMax < 0) return null;
  // Local-space distance is only used for ordering; scale can differ per instance, so
  // convert the hit point back to world and measure there.
  const t = Math.max(tMin, 0);
  const localHit = vec3.addScaled(o, d, t);
  const world = mat4.invert(worldInverse);
  const worldHit = vec3.transformMat4(localHit, world);
  return vec3.distance(ray.origin, worldHit);
}

export function raySphere(ray: Ray, center: Float32Array | number[], radius: number): number | null {
  const oc = vec3.sub(ray.origin, center);
  const b = vec3.dot(oc, ray.direction);
  const c = vec3.dot(oc, oc) - radius * radius;
  const disc = b * b - c;
  if (disc < 0) return null;
  const t = -b - Math.sqrt(disc);
  return t >= 0 ? t : null;
}

/** Intersection with the horizontal plane y = height, or null when parallel/behind. */
export function rayPlaneY(ray: Ray, height: number): Float32Array | null {
  if (Math.abs(ray.direction[1]) < 1e-6) return null;
  const t = (height - ray.origin[1]) / ray.direction[1];
  if (t < 0) return null;
  return vec3.addScaled(ray.origin, ray.direction, t, vec3.create());
}

export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** Sequential tweens over named scalar properties. */
export interface Tween<K extends string> {
  prop: K;
  to: number;
  duration: number;
}

export class TweenQueue<K extends string> {
  private queue: Array<Tween<K> & { from?: number; elapsed: number }> = [];
  constructor(private readonly values: Record<K, number>) {}

  get busy(): boolean {
    return this.queue.length > 0;
  }

  set(tweens: Tween<K>[]): void {
    this.queue = tweens.map((t) => ({ ...t, elapsed: 0 }));
  }

  /** Advances the head tween; returns true when a value changed. */
  update(dt: number): boolean {
    const head = this.queue[0];
    if (!head) return false;
    if (head.from === undefined) head.from = this.values[head.prop];
    head.elapsed = Math.min(head.duration, head.elapsed + dt);
    const t = head.duration > 0 ? head.elapsed / head.duration : 1;
    this.values[head.prop] = head.from + (head.to - head.from) * easeInOutCubic(t);
    if (head.elapsed >= head.duration) {
      this.values[head.prop] = head.to;
      this.queue.shift();
    }
    return true;
  }
}

/** Distinguishes a tap from a drag on pointer up. */
export function isTap(down: { x: number; y: number; time: number }, up: { x: number; y: number; time: number }): boolean {
  return up.time - down.time < 350 && Math.hypot(up.x - down.x, up.y - down.y) < 6;
}
