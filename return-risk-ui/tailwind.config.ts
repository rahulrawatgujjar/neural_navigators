import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",        // ✅ for app router pages
    "./components/**/*.{js,ts,jsx,tsx}" // ✅ THIS MATCHES YOUR REAL STRUCTURE
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};

export default config;
