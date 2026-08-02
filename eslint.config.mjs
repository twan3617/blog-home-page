import { defineConfig, globalIgnores } from 'eslint/config'
import nextVitals from 'eslint-config-next/core-web-vitals'
import nextTypeScript from 'eslint-config-next/typescript'

export default defineConfig([
  ...nextVitals,
  ...nextTypeScript,
  globalIgnores([
    '.next/**',
    'coverage/**',
    'components/AboutMe.js',
    'components/BlogSection.tsx',
    'components/CareerTimeline.js',
    'components/CombinedPanel.js',
    'components/CombinedTopPanel.tsx',
    'components/ConnectPanel.tsx',
    'components/EducationTimeline.js',
    'components/Footer.js',
    'components/layout.tsx',
    'global.d.ts',
    'pages/**',
  ]),
])
