/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Warm paper canvas — no gradients, no neon
        paper: '#F6F4EF',
        card: '#FFFFFF',
        edge: {
          DEFAULT: '#E6E1D6',   // hairline borders
          strong: '#D8D2C4',
          soft: '#EFEBE2',
        },
        ink: {
          DEFAULT: '#23201B',   // primary text
          soft: '#5D574B',      // secondary text
          faint: '#9A9284',     // tertiary/meta text
        },
        // Single interactive accent: forest green
        accent: {
          DEFAULT: '#2F6B4F',
          deep: '#255743',
          wash: '#EAF1EC',
          edge: '#C9DCD1',
        },
        // Severity
        danger: {
          DEFAULT: '#B3261E',
          deep: '#8F1D17',
          wash: '#FBEDEC',
          edge: '#EBCFCB',
        },
        caution: {
          DEFAULT: '#946200',
          wash: '#FBF3E2',
          edge: '#EBDDBB',
        },
        // Agent identities (muted, distinguishable)
        agent: {
          planner: '#704826',   // russet
          analyst: '#2F6B4F',   // forest
          ml: '#B0562F',        // terracotta
          reporter: '#8A7A2E',  // olive gold
        },
        // Live console panel (the one intentionally dark surface)
        walnut: {
          950: '#1C1915',
          900: '#242019',
          800: '#2E2921',
          700: '#3A342A',
          text: '#EFEAE0',
          mute: '#A79E8F',
        },
      },
      fontFamily: {
        sans: ['"Instrument Sans"', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        serif: ['"Source Serif 4"', 'Georgia', 'Cambria', 'serif'],
        code: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(33,29,24,0.04), 0 2px 10px rgba(33,29,24,0.05)',
        pop: '0 2px 4px rgba(33,29,24,0.06), 0 12px 32px rgba(33,29,24,0.12)',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.35' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.25s ease-out both',
        blink: 'blink 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
