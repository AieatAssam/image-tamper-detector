import { ChakraProvider } from '@chakra-ui/react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { system } from '../theme';
import type { AnalysisResponse, DetectorInfo, DetectorResult } from '../types/api';
import AnalysisResults from './AnalysisResults';

afterEach(cleanup);

const detectors: DetectorInfo[] = [
  {
    id: 'strong',
    name: 'Strong detector',
    family: 'test',
    applicable_formats: ['jpeg'],
    produces_map: true,
    description: 'A test detector.',
    limitations: [],
    enabled: true,
  },
  {
    id: 'weak',
    name: 'Weak detector',
    family: 'test',
    applicable_formats: ['jpeg'],
    produces_map: true,
    description: 'Another test detector.',
    limitations: [],
    enabled: true,
  },
  {
    id: 'metadata',
    name: 'Metadata detector',
    family: 'test',
    applicable_formats: ['jpeg'],
    produces_map: false,
    description: 'Not applicable in this fixture.',
    limitations: [],
    enabled: true,
  },
];

function result(
  id: string,
  score: number | null,
  state: DetectorResult['state'] = 'applicable',
): DetectorResult {
  return {
    id,
    state,
    flagged: score === null ? null : score >= 0.5,
    score,
    threshold: 0.5,
    reason: state === 'not_applicable' ? 'JPEG metadata is absent.' : 'Test result.',
    metrics: id === 'strong' ? { hanley_mcneil_se: 0.09 } : {},
    visualization_png_base64: id === 'metadata' ? null : 'ZmFrZQ==',
    duration_ms: 1,
    error: null,
  };
}

const results: AnalysisResponse = {
  verdict: 'inconclusive',
  score: 0.54,
  summary: 'The test evidence is mixed.',
  image: { width: 100, height: 80, format: 'JPEG', bytes: 10, sha256: 'test' },
  detectors: [
    result('weak', 0.54),
    result('strong', 0.72),
    result('metadata', null, 'not_applicable'),
  ],
  fusion: {
    method: 'weighted_logit',
    contributions: [{ id: 'strong', weight: 0.2, signed_contribution: 0.1 }],
    calibration_version: 'test',
  },
  warnings: [],
};

function renderResults() {
  return render(
    <ChakraProvider value={system}>
      <AnalysisResults results={results} originalUrl="blob:test" detectors={detectors} />
    </ChakraProvider>,
  );
}

describe('AnalysisResults', () => {
  it('orders score dots by magnitude and keeps NOT_APPLICABLE out of the plot', () => {
    const { container } = renderResults();
    expect(
      [...container.querySelectorAll('[data-detector-id]')].map((node) =>
        node.getAttribute('data-detector-id'),
      ),
    ).toEqual(['strong', 'weak']);
    expect(container.querySelector('[data-uncertainty="0.09"]')).toBeTruthy();
    expect(screen.getAllByText('NOT_APPLICABLE')).toHaveLength(2);
    expect(screen.getByRole('table')).toBeTruthy();
  });

  it('keeps the viewport controls when switching maps', () => {
    renderResults();
    fireEvent.change(screen.getByLabelText('Overlay zoom'), { target: { value: '2.4' } });
    fireEvent.change(screen.getByLabelText('Evidence layer'), { target: { value: 'weak' } });
    expect(screen.getByText('2.4×')).toBeTruthy();
    expect(screen.getByText(/Edge mask · Weak detector/)).toBeTruthy();
  });

  it('uses a labelled inconclusive verdict instead of color alone', () => {
    renderResults();
    const status = screen.getByRole('status');
    expect(status.textContent).toContain('Inconclusive evidence');
    expect(status.textContent).toContain('?');
    expect(status.closest('[data-verdict]')?.getAttribute('data-verdict')).toBe('inconclusive');
  });
});
