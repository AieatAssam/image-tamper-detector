import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import App from './App';

vi.mock('./api/client', () => ({
  getDetectors: vi.fn().mockResolvedValue({ detectors: [] }),
  validateImageDimensions: vi.fn().mockResolvedValue(undefined),
  analyze: vi.fn(),
  errorMessage: vi.fn(() => 'error'),
  AnalysisError: class AnalysisError extends Error {},
}));

describe('app', () => {
  it('renders the upload surface and disclaimer', async () => {
    render(<App />);
    expect(screen.getByText('Inspect the evidence')).toBeTruthy();
    expect(screen.getByText(/not proof/i)).toBeTruthy();
  });
});
