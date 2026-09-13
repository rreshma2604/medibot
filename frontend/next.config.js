/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    // Where the FastAPI backend lives. Change here if you move the port.
    NEXT_PUBLIC_API: process.env.NEXT_PUBLIC_API || "http://localhost:8000",
  },
};
module.exports = nextConfig;
