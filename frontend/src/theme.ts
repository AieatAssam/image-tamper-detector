import { createSystem, defaultConfig } from '@chakra-ui/react';

export const system = createSystem(defaultConfig, {
  theme: {
    tokens: {
      fonts: {
        body: { value: "'Avenir Next', 'Segoe UI', sans-serif" },
        heading: { value: "'Avenir Next', 'Segoe UI', sans-serif" },
        mono: { value: "'SFMono-Regular', Consolas, monospace" },
      },
      colors: {
        ink: { value: '#14212b' },
        muted: { value: '#60707b' },
        surface: { value: '#f5f7f8' },
        surfaceRaised: { value: '#ffffff' },
        line: { value: '#dce4e8' },
        signal: { value: '#c85b32' },
        signalSoft: { value: '#f7e4dc' },
        trustworthy: { value: '#28745b' },
        caution: { value: '#a36c1f' },
        uncertain: { value: '#53627a' },
      },
    },
    semanticTokens: {
      colors: {
        ink: { value: { base: '{colors.ink}', _dark: '#e8eef1' } },
        muted: { value: { base: '{colors.muted}', _dark: '#aebbc2' } },
        surface: { value: { base: '{colors.surface}', _dark: '#14212b' } },
        surfaceRaised: { value: { base: '{colors.surfaceRaised}', _dark: '#1c2d38' } },
        line: { value: { base: '{colors.line}', _dark: '#334955' } },
      },
    },
  },
  globalCss: {
    'html, body, #root': { minHeight: '100%' },
    body: { bg: 'surface', color: 'ink', margin: 0 },
  },
});
