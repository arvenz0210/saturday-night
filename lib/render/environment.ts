// Builds the image-based lighting set at startup, entirely on the GPU:
//   1. procedural studio environment  -> HDR equirect with mips
//   2. GGX prefiltered specular chain  -> one mip level per roughness step
//   3. cosine-convolved irradiance map
//   4. split-sum BRDF lookup table
import { effect, frame, sampler, target, type Gpu, type Target, type Texture } from "vgpu";
import { Blitter, mipLevelCount } from "./textures";
import envShader from "./shaders/env.wgsl";
import prefilterShader from "./shaders/prefilter.wgsl";
import irradianceShader from "./shaders/irradiance.wgsl";
import brdfShader from "./shaders/brdf.wgsl";

export interface Environment {
  /** Prefiltered radiance, mip i = roughness i / (specularMips - 1). */
  specular: Texture;
  specularMips: number;
  irradiance: Target;
  brdfLut: Target;
  sampler: GPUSampler;
  lutSampler: GPUSampler;
  dispose(): void;
}

const ENV_WIDTH = 1024;
const ENV_HEIGHT = 512;
const SPECULAR_WIDTH = 512;
const SPECULAR_HEIGHT = 256;
const SPECULAR_MIPS = 7;

export async function buildEnvironment(gpu: Gpu, blitter: Blitter, quality: "high" | "mobile" = "high"): Promise<Environment> {
  const sampleScale = quality === "mobile" ? 0.4 : 1;
  const device = gpu.gpu;
  const envSampler = sampler(gpu, {
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge",
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
  });
  const lutSampler = sampler(gpu, { addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", magFilter: "linear", minFilter: "linear" });

  // 1. Environment radiance -------------------------------------------------------------
  const envTarget = target(gpu, { size: [ENV_WIDTH, ENV_HEIGHT], format: "rgba16float", label: "env" });
  const envEffect = effect(gpu, envShader, { label: "studio-env" });
  await envEffect.compile(envTarget);
  envEffect.draw(envTarget);

  const envMips = mipLevelCount(ENV_WIDTH, ENV_HEIGHT);
  const envTexture = gpu.device.createTexture({
    label: "env-mipped",
    size: [ENV_WIDTH, ENV_HEIGHT],
    format: "rgba16float",
    mipLevelCount: envMips,
    usage: ["texture_binding", "render_attachment", "copy_dst"],
  });
  {
    const encoder = device.createCommandEncoder({ label: "env-mips" });
    blitter.blit(encoder, envTarget.color.createView(), envTexture.gpu.createView({ baseMipLevel: 0, mipLevelCount: 1 }), "rgba16float");
    blitter.generateMipmaps(encoder, envTexture.gpu);
    device.queue.submit([encoder.finish()]);
  }

  // 2. Prefiltered specular chain --------------------------------------------------------
  const specular = gpu.device.createTexture({
    label: "env-specular",
    size: [SPECULAR_WIDTH, SPECULAR_HEIGHT],
    format: "rgba16float",
    mipLevelCount: SPECULAR_MIPS,
    usage: ["texture_binding", "render_attachment", "copy_dst"],
  });
  const prefilter = effect(gpu, prefilterShader, {
    label: "prefilter",
    set: {
      params: { roughness: 0, sampleCount: 96, srcWidth: ENV_WIDTH, srcHeight: ENV_HEIGHT, srcMips: envMips },
      env: envTexture,
      envSampler,
    },
  });
  const levelTargets: Target[] = [];
  for (let level = 0; level < SPECULAR_MIPS; level++) {
    const w = Math.max(1, SPECULAR_WIDTH >> level);
    const h = Math.max(1, SPECULAR_HEIGHT >> level);
    const levelTarget = target(gpu, { size: [w, h], format: "rgba16float", label: `prefilter-${level}` });
    levelTargets.push(levelTarget);
    const roughness = level / (SPECULAR_MIPS - 1);
    // Rougher levels are smaller, so they can afford more samples.
    const sampleCount = level === 0 ? 1 : Math.round(Math.min(512, 96 + level * 72) * sampleScale);
    prefilter.set({ params: { roughness, sampleCount, srcWidth: ENV_WIDTH, srcHeight: ENV_HEIGHT, srcMips: envMips } });
    frame(gpu, (f) => f.pass(levelTarget, prefilter));
    const encoder = device.createCommandEncoder({ label: `prefilter-copy-${level}` });
    blitter.blit(encoder, levelTarget.color.createView(), specular.gpu.createView({ baseMipLevel: level, mipLevelCount: 1 }), "rgba16float");
    device.queue.submit([encoder.finish()]);
  }

  // 3. Irradiance -------------------------------------------------------------------------
  const irradiance = target(gpu, { size: [128, 64], format: "rgba16float", label: "irradiance" });
  const irradianceEffect = effect(gpu, irradianceShader, {
    label: "irradiance",
    set: { params: { sampleCount: Math.round(384 * sampleScale), lod: 3 }, env: envTexture, envSampler },
  });
  frame(gpu, (f) => f.pass(irradiance, irradianceEffect));

  // 4. BRDF LUT ------------------------------------------------------------------------------
  const brdfLut = target(gpu, { size: [256, 256], format: "rg16float", label: "brdf-lut" });
  const brdfEffect = effect(gpu, brdfShader, { label: "brdf-lut" });
  frame(gpu, (f) => f.pass(brdfLut, brdfEffect));

  return {
    specular,
    specularMips: SPECULAR_MIPS,
    irradiance,
    brdfLut,
    sampler: envSampler,
    lutSampler,
    dispose() {
      specular.destroy();
      envTexture.destroy();
      for (const t of levelTargets) t.color.destroy();
      envTarget.color.destroy();
    },
  };
}
