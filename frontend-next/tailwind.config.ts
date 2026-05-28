import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#030712",
        panel: "rgba(10, 18, 34, 0.66)",
        line: "rgba(148, 163, 184, 0.18)",
        cyanGlow: "#38bdf8",
        violetGlow: "#8b5cf6",
        bull: "#34d399",
        bear: "#fb7185",
      },
      boxShadow: {
        glass: "0 22px 80px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.08)",
        glow: "0 0 36px rgba(56,189,248,.24)",
        violet: "0 0 42px rgba(139,92,246,.22)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "ui-sans-serif", "system-ui"],
      },
      keyframes: {
        pulseRing: {
          "0%, 100%": { opacity: ".28", transform: "scale(.96)" },
          "50%": { opacity: ".64", transform: "scale(1.04)" },
        },
        drift: {
          "0%": { transform: "translate3d(0,0,0)" },
          "50%": { transform: "translate3d(18px,-20px,0)" },
          "100%": { transform: "translate3d(0,0,0)" },
        },
      },
      animation: {
        pulseRing: "pulseRing 3.5s ease-in-out infinite",
        drift: "drift 8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
