import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: path.join(__dirname),
  // Type and lint errors fail the build. The previous settings hid nine type errors (audit F5).
};

export default nextConfig;
