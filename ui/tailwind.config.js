/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        charcoal: {
          900: "#0a0a0a",
          800: "#141414",
          700: "#1f1f1f",
          600: "#2a2a2a",
          500: "#3d3d3d",
        },
        amber: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
        sans: ["Instrument Sans", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
