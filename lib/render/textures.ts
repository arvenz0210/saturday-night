// Texture helpers on top of vgpu's core Device: image upload with full mip chains,
// a tiny blit pipeline (used for mipmaps and for filling mip levels of the
// prefiltered environment), and solid fallback textures.
import type { Gpu, Texture } from "vgpu";

const BLIT_WGSL = /* wgsl */ `
  struct VSOut { @builtin(position) position: vec4f, @location(0) uv: vec2f }
  @vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    var out: VSOut;
    out.position = vec4f(p[vi], 0.0, 1.0);
    out.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5);
    return out;
  }
  @group(0) @binding(0) var src: texture_2d<f32>;
  @group(0) @binding(1) var samp: sampler;
  @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSampleLevel(src, samp, uv, 0.0);
  }
`;

export class Blitter {
  private readonly pipelines = new Map<GPUTextureFormat, GPURenderPipeline>();
  private readonly module: GPUShaderModule;
  private readonly sampler: GPUSampler;

  constructor(private readonly device: GPUDevice) {
    this.module = device.createShaderModule({ label: "blit", code: BLIT_WGSL });
    this.sampler = device.createSampler({ minFilter: "linear", magFilter: "linear" });
  }

  private pipeline(format: GPUTextureFormat): GPURenderPipeline {
    let p = this.pipelines.get(format);
    if (!p) {
      p = this.device.createRenderPipeline({
        label: `blit-${format}`,
        layout: "auto",
        vertex: { module: this.module, entryPoint: "vs_main" },
        fragment: { module: this.module, entryPoint: "fs_main", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
      this.pipelines.set(format, p);
    }
    return p;
  }

  /** Encodes a full-screen copy (with bilinear filtering) from src view into dst view. */
  blit(encoder: GPUCommandEncoder, src: GPUTextureView, dst: GPUTextureView, format: GPUTextureFormat): void {
    const pipeline = this.pipeline(format);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src },
        { binding: 1, resource: this.sampler },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: dst, loadOp: "clear", clearValue: [0, 0, 0, 0], storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }

  /** Generates levels 1..n of a 2D texture from level 0. */
  generateMipmaps(encoder: GPUCommandEncoder, texture: GPUTexture): void {
    for (let level = 1; level < texture.mipLevelCount; level++) {
      const src = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 });
      const dst = texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
      this.blit(encoder, src, dst, texture.format);
    }
  }
}

export function mipLevelCount(width: number, height: number): number {
  return Math.floor(Math.log2(Math.max(width, height))) + 1;
}

export interface ImageTextureOptions {
  srgb: boolean;
  label?: string;
}

/** Decodes an encoded image (PNG/JPEG/WebP bytes) and uploads it with a full mip chain. */
export async function createImageTexture(
  gpu: Gpu,
  blitter: Blitter,
  bytes: Uint8Array,
  mimeType: string,
  opts: ImageTextureOptions,
): Promise<Texture> {
  const blob = new Blob([bytes as BlobPart], { type: mimeType });
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  try {
    const { width, height } = bitmap;
    const format: GPUTextureFormat = opts.srgb ? "rgba8unorm-srgb" : "rgba8unorm";
    const texture = gpu.device.createTexture({
      label: opts.label,
      size: [width, height],
      format,
      mipLevelCount: mipLevelCount(width, height),
      usage: ["texture_binding", "copy_dst", "render_attachment"],
    });
    gpu.gpu.queue.copyExternalImageToTexture(
      { source: bitmap, flipY: false },
      { texture: texture.gpu, mipLevel: 0 },
      [width, height],
    );
    const encoder = gpu.gpu.createCommandEncoder({ label: "mipmaps" });
    blitter.generateMipmaps(encoder, texture.gpu);
    gpu.gpu.queue.submit([encoder.finish()]);
    return texture;
  } finally {
    bitmap.close();
  }
}

/** 1x1 texture with a constant color (values 0..1). */
export function createSolidTexture(gpu: Gpu, rgba: [number, number, number, number], label?: string): Texture {
  const texture = gpu.device.createTexture({
    label,
    size: [1, 1],
    format: "rgba8unorm",
    usage: ["texture_binding", "copy_dst"],
  });
  const data = new Uint8Array(rgba.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255)));
  gpu.gpu.queue.writeTexture({ texture: texture.gpu }, data, { bytesPerRow: 4 }, [1, 1]);
  return texture;
}

/** glTF sampler enums -> WebGPU sampler descriptor. Anisotropy needs trilinear filtering. */
export function gltfSamplerDescriptor(sampler?: { magFilter?: number; minFilter?: number; wrapS: number; wrapT: number }): GPUSamplerDescriptor {
  const wrap = (mode: number): GPUAddressMode =>
    mode === 33071 ? "clamp-to-edge" : mode === 33648 ? "mirror-repeat" : "repeat";
  const nearestMag = sampler?.magFilter === 9728;
  const nearestMin = sampler?.minFilter === 9728 || sampler?.minFilter === 9984 || sampler?.minFilter === 9986;
  const trilinear = !nearestMag && !nearestMin;
  return {
    addressModeU: wrap(sampler?.wrapS ?? 10497),
    addressModeV: wrap(sampler?.wrapT ?? 10497),
    magFilter: nearestMag ? "nearest" : "linear",
    minFilter: nearestMin ? "nearest" : "linear",
    mipmapFilter: "linear",
    maxAnisotropy: trilinear ? 16 : 1,
  };
}
