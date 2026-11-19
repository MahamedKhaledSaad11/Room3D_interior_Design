import type { NextConfig } from "next";

const backendOrigin = process.env.ROOM3D_BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  distDir: process.env.ROOM3D_NEXT_DIST_DIR ?? ".next",
  devIndicators: process.env.ROOM3D_NEXT_DIST_DIR ? false : undefined,
  reactStrictMode: true,
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  experimental: {
    middlewareClientMaxBodySize: "500mb",
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendOrigin}/api/:path*` }];
  },
};

export default nextConfig;
