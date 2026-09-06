import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // vgpu WGSL module loader: lets .wgsl files `import`/`export` between each other
  // and hands vgpu one resolved ShaderSource per entry shader.
  turbopack: {
    rules: {
      "*.wgsl": {
        loaders: ["@vgpu/wgsl/loader-webpack"],
        as: "*.js",
      },
    },
  },
  webpack(config) {
    config.module ??= {};
    config.module.rules ??= [];
    config.module.rules.push({
      test: /\.wgsl$/,
      loader: "@vgpu/wgsl/loader-webpack",
    });
    return config;
  },
};

export default nextConfig;
