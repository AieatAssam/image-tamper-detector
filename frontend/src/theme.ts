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
        ink: { value: 'var(--ink)' },
        muted: { value: 'var(--muted)' },
        surface: { value: 'var(--surface)' },
        surfaceRaised: { value: 'var(--surface-raised)' },
        line: { value: 'var(--line)' },
        signal: { value: 'var(--status-manipulated)' },
        signalSoft: { value: 'var(--status-manipulated-soft)' },
        trustworthy: { value: 'var(--status-authentic)' },
        caution: { value: 'var(--status-caution)' },
        uncertain: { value: 'var(--status-inconclusive)' },
      },
    },
    semanticTokens: {
      colors: {
        ink: { value: '{colors.ink}' },
        muted: { value: '{colors.muted}' },
        surface: { value: '{colors.surface}' },
        surfaceRaised: { value: '{colors.surfaceRaised}' },
        line: { value: '{colors.line}' },
      },
    },
  },
  globalCss: {
    'html, body, #root': { minHeight: '100%' },
    body: { bg: 'surface', color: 'ink', margin: 0 },
  },
});
