/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // FUI Blueprint Color Palette
      colors: {
        'editorial': {
          'deep': '#050506',
          'base': '#0a0a0c',
          'elevated': '#121218',
          'purple': '#8B5CF6',
          'pink': '#EC4899',
          'blue': '#1E3A8A',
          'violet': '#7C3AED',
          'lavender': '#A78BFA',
        },
        'fui': {
          'black': '#000000',
          'dark': '#0a0a0a',
          'panel': '#111111',
          'gray': {
            100: 'var(--fui-gray-100, #E0E0E0)',
            200: 'var(--fui-gray-200, #CCCCCC)',
            300: 'var(--fui-gray-300, #999999)',
            400: 'var(--fui-gray-400, #888888)',
            500: 'var(--fui-gray-500, #666666)',
            600: 'var(--fui-gray-600, #444444)',
            700: 'var(--fui-gray-700, #333333)',
            800: 'var(--fui-gray-800, #222222)',
            900: 'var(--fui-gray-900, #111111)',
          },
          'accent': {
            'cyan': '#00FFFF',
            'blue': '#4488FF',
            'green': '#00FF88',
            'orange': '#FFAA00',
            'red': '#FF4444',
            'critical': '#FF0044',
          }
        },
        // Theme-aware semantic colors (uses CSS variables)
        'theme': {
          'bg': {
            'primary': 'var(--bg-primary)',
            'secondary': 'var(--bg-secondary)',
            'tertiary': 'var(--bg-tertiary)',
            'panel': 'var(--bg-panel)',
            'elevated': 'var(--bg-panel-elevated)',
          },
          'text': {
            'primary': 'var(--text-primary)',
            'secondary': 'var(--text-secondary)',
            'muted': 'var(--text-muted)',
            'dim': 'var(--text-dim)',
          },
          'border': {
            'DEFAULT': 'var(--border-color)',
            'highlight': 'var(--border-highlight)',
            'dim': 'var(--border-dim)',
          },
          'accent': {
            'primary': 'var(--accent-primary)',
            'cyan': 'var(--accent-cyan)',
            'blue': 'var(--accent-blue)',
            'emerald': 'var(--accent-emerald)',
          },
          'risk': {
            'low': 'var(--risk-low)',
            'medium': 'var(--risk-medium)',
            'high': 'var(--risk-high)',
            'critical': 'var(--risk-critical)',
          }
        }
      },
      // Editorial triple-stack font families
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'Roboto Mono', 'monospace'],
        'display': ['Instrument Serif', 'Georgia', 'serif'],
        'cursive': ['Cormorant Garamond', 'Georgia', 'serif'],
        'heading': ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        'body': ['Inter', 'system-ui', 'sans-serif'],
      },
      // Hairline borders
      borderWidth: {
        'hairline': '1px',
      },
      // Blueprint grid background
      backgroundImage: {
        'grid-blueprint': `
          linear-gradient(rgba(102, 102, 102, 0.15) 1px, transparent 1px),
          linear-gradient(90deg, rgba(102, 102, 102, 0.15) 1px, transparent 1px)
        `,
        'grid-fine': `
          linear-gradient(rgba(102, 102, 102, 0.08) 1px, transparent 1px),
          linear-gradient(90deg, rgba(102, 102, 102, 0.08) 1px, transparent 1px)
        `,
      },
      backgroundSize: {
        'grid-40': '40px 40px',
        'grid-8': '8px 8px',
      },
      // FUI Shadows/Glows
      boxShadow: {
        'fui': '0 0 20px rgba(224, 224, 224, 0.1)',
        'fui-cyan': '0 0 15px rgba(0, 255, 255, 0.2)',
        'fui-glow': 'inset 0 0 0 1px rgba(102, 102, 102, 0.2), 0 0 30px rgba(0, 0, 0, 0.5)',
        'theme': 'var(--glow-primary)',
        'theme-accent': 'var(--glow-accent)',
      },
      // Animation timing
      animation: {
        'flicker': 'flicker 4s ease-in-out infinite',
        'scan': 'scan 3s linear infinite',
      },
      keyframes: {
        flicker: {
          '0%, 100%': { opacity: '1' },
          '92%': { opacity: '1' },
          '93%': { opacity: '0.8' },
          '94%': { opacity: '1' },
          '96%': { opacity: '0.9' },
          '97%': { opacity: '1' },
        },
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        }
      },
      // Letter spacing for technical labels
      letterSpacing: {
        'technical': '0.15em',
        'wide-technical': '0.2em',
      }
    },
  },
  plugins: [],
}