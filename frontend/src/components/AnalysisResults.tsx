import { Box, Card, Heading, HStack, Text, VStack } from '@chakra-ui/react';
import type { KeyboardEvent, PointerEvent, WheelEvent } from 'react';
import { useEffect, useRef, useState } from 'react';
import type {
  AnalysisResponse,
  DetectorInfo,
  DetectorResult,
  DetectorState,
  Verdict,
} from '../types/api';

interface Props {
  results: AnalysisResponse;
  originalUrl: string;
  detectors: DetectorInfo[];
}

type ResultWithUncertainty = DetectorResult & {
  standard_error?: number;
  hanley_mcneil_se?: number;
};

const SCORE_THRESHOLD = 0.5;

const verdictCopy: Record<
  Verdict,
  {
    title: string;
    description: string;
    tone: 'manipulated' | 'authentic' | 'inconclusive';
    icon: string;
  }
> = {
  manipulated: {
    title: 'Strong manipulation signal',
    description: 'Several available signals point toward a manipulated image.',
    tone: 'manipulated',
    icon: '▲',
  },
  likely_manipulated: {
    title: 'Likely manipulated',
    description: 'The evidence leans toward image manipulation, but it is not proof.',
    tone: 'manipulated',
    icon: '↗',
  },
  inconclusive: {
    title: 'Inconclusive evidence',
    description: 'There is not enough applicable evidence for a reliable direction.',
    tone: 'inconclusive',
    icon: '?',
  },
  likely_authentic: {
    title: 'Likely authentic',
    description: 'The available evidence leans toward an authentic image.',
    tone: 'authentic',
    icon: '✓',
  },
  authentic: {
    title: 'Authentic signal',
    description: 'The available signals support an authentic image.',
    tone: 'authentic',
    icon: '✓',
  },
};

const stateCopy: Record<DetectorState, { label: string; icon: string }> = {
  applicable: { label: 'Applicable', icon: '●' },
  not_applicable: { label: 'NOT_APPLICABLE', icon: '—' },
  error: { label: 'Error', icon: '!' },
};

function detectorName(id: string, detectors: DetectorInfo[]): string {
  return detectors.find((detector) => detector.id === id)?.name ?? id.replaceAll('_', ' ');
}

function scoreText(score: number | null): string {
  return score === null ? '—' : `${Math.round(score * 100)}%`;
}

function metricLabel(key: string): string {
  if (key === 'auc') return 'training AUC';
  if (key === 'auc_standard_error') return 'training AUC standard error';
  return key.replaceAll('_', ' ');
}

function uncertaintyFor(result: DetectorResult): number | null {
  const candidate = result as ResultWithUncertainty;
  const value =
    candidate.standard_error ??
    candidate.hanley_mcneil_se ??
    result.metrics.hanley_mcneil_se ??
    result.metrics.auc_standard_error ??
    result.metrics.standard_error;
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? Math.min(value, 1)
    : null;
}

function stateLabel(result: DetectorResult): string {
  if (result.state === 'applicable') {
    return result.flagged === true
      ? 'Above detector threshold'
      : result.flagged === false
        ? 'Below detector threshold'
        : 'Signal available';
  }
  return result.state === 'not_applicable'
    ? 'No valid comparison for this image'
    : 'Detector failed';
}

function ScoreRuler({
  score,
  threshold,
  label,
}: {
  score: number;
  threshold: number;
  label: string;
}) {
  return (
    <Box className="score-ruler" aria-label={`${label}: ${scoreText(score)}`}>
      <HStack justify="space-between" mb={2}>
        <Text fontSize="sm" color="muted">
          {label}
        </Text>
        <Text fontFamily="mono" fontSize="lg" fontWeight="bold">
          {scoreText(score)}
        </Text>
      </HStack>
      <Box className="score-ruler-track" aria-hidden="true">
        <Box className="score-ruler-threshold" left={`${threshold * 100}%`} />
        <Box className="score-ruler-marker" left={`${score * 100}%`} />
      </Box>
      <HStack justify="space-between" mt={2} fontFamily="mono" fontSize="xs" color="muted">
        <Text>0%</Text>
        <Text>threshold {Math.round(threshold * 100)}%</Text>
        <Text>100%</Text>
      </HStack>
    </Box>
  );
}

function StatePill({ state }: { state: DetectorState }) {
  const copy = stateCopy[state];
  return (
    <span className={`state-pill state-pill--${state}`}>
      <span aria-hidden="true">{copy.icon}</span>
      {copy.label}
    </span>
  );
}

function DetectorTable({
  results,
  detectors,
}: {
  results: DetectorResult[];
  detectors: DetectorInfo[];
}) {
  return (
    <Box className="table-wrap">
      <table className="detector-table">
        <caption className="sr-only">
          Detector scores, training-time uncertainty, and applicability
        </caption>
        <thead>
          <tr>
            <th scope="col">Detector</th>
            <th scope="col">State</th>
            <th scope="col">Score</th>
            <th scope="col">± training SE</th>
            <th scope="col">Threshold</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result) => {
            const uncertainty = uncertaintyFor(result);
            return (
              <tr key={result.id} data-state={result.state}>
                <th scope="row">
                  <span className="table-detector-name">{detectorName(result.id, detectors)}</span>
                  <span className="table-detector-note">{stateLabel(result)}</span>
                </th>
                <td>
                  <StatePill state={result.state} />
                </td>
                <td className="table-number">
                  {result.score === null ? '—' : scoreText(result.score)}
                </td>
                <td className="table-number">
                  {uncertainty === null ? 'Not returned' : `±${Math.round(uncertainty * 100)}%`}
                </td>
                <td className="table-number">{scoreText(result.threshold)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Box>
  );
}

function ScoreDotPlot({
  results,
  detectors,
}: {
  results: DetectorResult[];
  detectors: DetectorInfo[];
}) {
  const points = results
    .filter(
      (result): result is DetectorResult & { score: number } =>
        result.state === 'applicable' && result.score !== null,
    )
    .sort((a, b) => b.score - a.score);
  const width = 760;
  const left = 174;
  const right = 34;
  const top = 30;
  const rowHeight = 44;
  const plotWidth = width - left - right;
  const height = Math.max(150, top + points.length * rowHeight + 30);
  const x = (value: number) => left + Math.max(0, Math.min(1, value)) * plotWidth;

  return (
    <Box className="dot-plot-wrap">
      {points.length ? (
        <svg
          className="dot-plot"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Detector scores with one training-time standard-error whiskers"
        >
          <title>Detector scores with training-time uncertainty</title>
          <line
            className="plot-threshold"
            x1={x(SCORE_THRESHOLD)}
            x2={x(SCORE_THRESHOLD)}
            y1={top - 12}
            y2={height - 28}
          />
          <text className="plot-threshold-label" x={x(SCORE_THRESHOLD) + 6} y={top - 15}>
            50% threshold
          </text>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
            <g key={tick}>
              <line className="plot-grid" x1={x(tick)} x2={x(tick)} y1={top - 4} y2={height - 28} />
              <text className="plot-axis-label" x={x(tick)} y={height - 8} textAnchor="middle">
                {Math.round(tick * 100)}%
              </text>
            </g>
          ))}
          {points.map((point, index) => {
            const y = top + index * rowHeight + 16;
            const uncertainty = uncertaintyFor(point);
            const low = uncertainty === null ? point.score : Math.max(0, point.score - uncertainty);
            const high =
              uncertainty === null ? point.score : Math.min(1, point.score + uncertainty);
            const label = detectorName(point.id, detectors);
            const detail =
              uncertainty === null
                ? 'training-time uncertainty not returned'
                : `plus or minus ${Math.round(uncertainty * 100)} percent training SE`;
            return (
              <g key={point.id} data-detector-id={point.id} data-score={point.score}>
                <title>{`${label}: ${scoreText(point.score)}; ${detail}`}</title>
                <text className="plot-detector-label" x={left - 12} y={y + 5} textAnchor="end">
                  {label}
                </text>
                {uncertainty !== null && (
                  <g data-uncertainty={uncertainty}>
                    <line className="plot-error" x1={x(low)} x2={x(high)} y1={y} y2={y} />
                    <line
                      className="plot-error-cap"
                      x1={x(low)}
                      x2={x(low)}
                      y1={y - 6}
                      y2={y + 6}
                    />
                    <line
                      className="plot-error-cap"
                      x1={x(high)}
                      x2={x(high)}
                      y1={y - 6}
                      y2={y + 6}
                    />
                  </g>
                )}
                <circle className="plot-dot" cx={x(point.score)} cy={y} r="6" />
                <text className="plot-score-label" x={x(point.score) + 12} y={y + 5}>
                  {scoreText(point.score)}
                </text>
              </g>
            );
          })}
        </svg>
      ) : (
        <Text color="muted">No applicable detector scores were returned for this image.</Text>
      )}
      <HStack className="plot-legend" gap={4} mt={2} wrap="wrap">
        <span className="plot-key">
          <span className="plot-key-dot" aria-hidden="true" /> Score
        </span>
        <span className="plot-key">
          <span className="plot-key-whisker" aria-hidden="true" /> ± one training SE
        </span>
        <Text fontSize="xs" color="muted">
          SE is estimated during detector training, not for this uploaded image. It is shown only
          when returned by the service.
        </Text>
      </HStack>
    </Box>
  );
}

function EvidenceCard({
  result,
  detectors,
}: {
  result: DetectorResult;
  detectors: DetectorInfo[];
}) {
  const info = detectors.find((detector) => detector.id === result.id);
  const uncertainty = uncertaintyFor(result);
  return (
    <Card.Root variant="outline" className="forensics-card" data-state={result.state}>
      <Card.Body>
        <details>
          <summary>
            <HStack display="inline-flex" gap={3} ml={2}>
              <Text fontWeight="bold">{info?.name ?? result.id}</Text>
              <StatePill state={result.state} />
            </HStack>
          </summary>
          <VStack align="stretch" gap={4} mt={4}>
            {info?.description && <Text color="muted">{info.description}</Text>}
            <Text>{result.error ?? result.reason}</Text>
            {result.score !== null && (
              <HStack gap={5} align="start" wrap="wrap">
                <Box>
                  <Text
                    fontSize="xs"
                    color="muted"
                    textTransform="uppercase"
                    letterSpacing="0.08em"
                  >
                    Score
                  </Text>
                  <Text fontFamily="mono" fontSize="xl" fontWeight="bold">
                    {scoreText(result.score)}
                  </Text>
                </Box>
                <Box>
                  <Text
                    fontSize="xs"
                    color="muted"
                    textTransform="uppercase"
                    letterSpacing="0.08em"
                  >
                    Training-time SE
                  </Text>
                  <Text fontFamily="mono" fontSize="xl" fontWeight="bold">
                    {uncertainty === null ? 'Not returned' : `±${Math.round(uncertainty * 100)}%`}
                  </Text>
                </Box>
                <Box>
                  <Text
                    fontSize="xs"
                    color="muted"
                    textTransform="uppercase"
                    letterSpacing="0.08em"
                  >
                    Threshold
                  </Text>
                  <Text fontFamily="mono" fontSize="xl" fontWeight="bold">
                    {scoreText(result.threshold)}
                  </Text>
                </Box>
              </HStack>
            )}
            {Object.keys(result.metrics).length > 0 && (
              <Box>
                <Text fontWeight="bold" mb={2}>
                  Measurements
                </Text>
                <VStack align="stretch" gap={1} fontFamily="mono" fontSize="sm">
                  {Object.entries(result.metrics).map(([key, value]) => (
                    <HStack key={key} justify="space-between">
                      <Text color="muted">{metricLabel(key)}</Text>
                      <Text>{value === null ? 'Not returned' : value.toFixed(3)}</Text>
                    </HStack>
                  ))}
                </VStack>
              </Box>
            )}
            {info?.limitations.length ? (
              <Box borderLeftWidth="3px" borderColor="line" pl={3}>
                <Text fontWeight="bold" fontSize="sm">
                  Limitations
                </Text>
                <VStack align="stretch" gap={1} mt={1}>
                  {info.limitations.map((limitation) => (
                    <Text key={limitation} fontSize="sm" color="muted">
                      {limitation}
                    </Text>
                  ))}
                </VStack>
              </Box>
            ) : null}
            <Text fontSize="xs" color="muted">
              Completed in {result.duration_ms} ms.
            </Text>
          </VStack>
        </details>
      </Card.Body>
    </Card.Root>
  );
}

function EdgeMask({ src, alt }: { src: string; alt: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let active = true;
    setReady(false);
    const image = new window.Image();
    image.onload = () => {
      if (!active || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      if (!context) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      context.drawImage(image, 0, 0);
      const source = context.getImageData(0, 0, canvas.width, canvas.height);
      const output = new Uint8ClampedArray(source.data.length);
      const luminance = (index: number) =>
        ((source.data[index] ?? 0) * 0.2126 +
          (source.data[index + 1] ?? 0) * 0.7152 +
          (source.data[index + 2] ?? 0) * 0.0722) /
        255;
      for (let y = 0; y < canvas.height; y += 1) {
        for (let x = 0; x < canvas.width; x += 1) {
          const index = (y * canvas.width + x) * 4;
          const here = luminance(index);
          const right = luminance(index + (x + 1 < canvas.width ? 4 : 0));
          const below = luminance(index + (y + 1 < canvas.height ? canvas.width * 4 : 0));
          const strength = Math.min(1, (Math.abs(here - right) + Math.abs(here - below)) * 4);
          const shade = Math.round(224 - strength * 146);
          output[index] = shade;
          output[index + 1] = Math.round(230 - strength * 145);
          output[index + 2] = 250;
          output[index + 3] = Math.round(strength * 235);
        }
      }
      context.putImageData(new ImageData(output, canvas.width, canvas.height), 0, 0);
      if (active) setReady(true);
    };
    image.src = src;
    return () => {
      active = false;
      image.onload = null;
    };
  }, [src]);

  return (
    <>
      <img
        className={`evidence-mask evidence-mask--fallback${ready ? ' is-ready' : ''}`}
        src={src}
        alt=""
        aria-hidden="true"
      />
      <canvas
        ref={canvasRef}
        className={`evidence-mask${ready ? ' is-ready' : ''}`}
        role="img"
        aria-label={alt}
        aria-hidden={!ready}
      />
    </>
  );
}

function OverlayViewer({ originalUrl, results, detectors }: Props) {
  const mapped = results.detectors.filter((result) => result.visualization_png_base64);
  const [selectedId, setSelectedId] = useState(mapped[0]?.id ?? '');
  const [opacity, setOpacity] = useState(78);
  const [divider, setDivider] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const viewportRef = useRef<HTMLDivElement>(null);
  const panStart = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);
  const dividerDragging = useRef(false);
  const selected = mapped.find((result) => result.id === selectedId) ?? mapped[0];

  useEffect(() => {
    if (!mapped.some((result) => result.id === selectedId)) setSelectedId(mapped[0]?.id ?? '');
  }, [mapped, selectedId]);

  if (!selected?.visualization_png_base64) return null;

  const updateDivider = (event: PointerEvent<HTMLDivElement>) => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    setDivider(
      Math.round(Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100))),
    );
  };
  const onDividerPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.stopPropagation();
    dividerDragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    updateDivider(event);
  };
  const onDividerPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dividerDragging.current) updateDivider(event);
  };
  const onDividerPointerUp = () => {
    dividerDragging.current = false;
  };
  const onDividerKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 10 : 5;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') {
      event.preventDefault();
      setDivider((value) => Math.max(0, value - step));
    } else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') {
      event.preventDefault();
      setDivider((value) => Math.min(100, value + step));
    } else if (event.key === 'Home') {
      event.preventDefault();
      setDivider(0);
    } else if (event.key === 'End') {
      event.preventDefault();
      setDivider(100);
    }
  };
  const onViewportPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    panStart.current = { x: event.clientX, y: event.clientY, panX: pan.x, panY: pan.y };
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onViewportPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!panStart.current) return;
    setPan({
      x: panStart.current.panX + event.clientX - panStart.current.x,
      y: panStart.current.panY + event.clientY - panStart.current.y,
    });
  };
  const onViewportPointerUp = () => {
    panStart.current = null;
  };
  const onViewportWheel = (event: WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    setZoom((value) => Math.max(1, Math.min(4, value + (event.deltaY > 0 ? -0.1 : 0.1))));
  };
  const move = (x: number, y: number) => setPan((value) => ({ x: value.x + x, y: value.y + y }));
  const transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;
  const mapUrl = `data:image/png;base64,${selected.visualization_png_base64}`;

  return (
    <Card.Root variant="outline" className="forensics-card">
      <Card.Body>
        <VStack align="stretch" gap={4}>
          <Box>
            <Text className="eyebrow">Compare / overlay</Text>
            <Heading size="md" mt={1}>
              Inspect the visual evidence
            </Heading>
            <Text color="muted" fontSize="sm" mt={1}>
              Drag the divider to compare the original with an edge-highlighted, single-hue detector
              mask. Drag the image to pan; scroll to zoom.
            </Text>
          </Box>
          <label className="field-label">
            <span>Evidence layer</span>
            <select
              className="native-select"
              value={selected.id}
              onChange={(event) => setSelectedId(event.target.value)}
              aria-label="Evidence layer"
            >
              {mapped.map((result) => (
                <option key={result.id} value={result.id}>
                  {detectorName(result.id, detectors)}
                </option>
              ))}
            </select>
          </label>
          <Box
            ref={viewportRef}
            className="comparison-viewport"
            role="group"
            aria-label="Original image and detector evidence comparison"
            onPointerDown={onViewportPointerDown}
            onPointerMove={onViewportPointerMove}
            onPointerUp={onViewportPointerUp}
            onPointerCancel={onViewportPointerUp}
            onWheel={onViewportWheel}
          >
            <Box className="comparison-layer" style={{ transform }}>
              <img
                className="comparison-image"
                src={originalUrl}
                alt="Original"
                draggable={false}
              />
            </Box>
            <Box
              className="comparison-layer comparison-evidence"
              style={{ clipPath: `inset(0 ${100 - divider}% 0 0)`, opacity: opacity / 100 }}
            >
              <Box className="comparison-layer" style={{ transform }}>
                <EdgeMask
                  src={mapUrl}
                  alt={`${detectorName(selected.id, detectors)} edge-highlighted map`}
                />
              </Box>
            </Box>
            <span className="viewport-label viewport-label--original">Original</span>
            <span className="viewport-label viewport-label--evidence">
              Edge mask · {detectorName(selected.id, detectors)}
            </span>
            <div
              className="swipe-divider"
              style={{ left: `${divider}%` }}
              role="slider"
              tabIndex={0}
              aria-label="Swipe divider"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={divider}
              aria-valuetext={`${divider}% original and ${100 - divider}% detector evidence`}
              onPointerDown={onDividerPointerDown}
              onPointerMove={onDividerPointerMove}
              onPointerUp={onDividerPointerUp}
              onPointerCancel={onDividerPointerUp}
              onKeyDown={onDividerKeyDown}
            >
              <span aria-hidden="true">↔</span>
            </div>
          </Box>
          <div className="comparison-controls">
            <label className="range-field">
              <HStack justify="space-between">
                <span>Mask opacity</span>
                <span className="control-value">{opacity}%</span>
              </HStack>
              <input
                type="range"
                min="0"
                max="100"
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
                aria-label="Mask opacity"
              />
            </label>
            <label className="range-field">
              <HStack justify="space-between">
                <span>Zoom</span>
                <span className="control-value">{zoom.toFixed(1)}×</span>
              </HStack>
              <input
                type="range"
                min="1"
                max="4"
                step="0.1"
                value={zoom}
                onChange={(event) => setZoom(Number(event.target.value))}
                aria-label="Overlay zoom"
              />
            </label>
            <fieldset className="pan-controls">
              <legend>Pan</legend>
              <button type="button" onClick={() => move(0, -24)} aria-label="Pan overlay up">
                ↑
              </button>
              <button type="button" onClick={() => move(-24, 0)} aria-label="Pan overlay left">
                ←
              </button>
              <button
                type="button"
                onClick={() => {
                  setPan({ x: 0, y: 0 });
                  setZoom(1);
                }}
                aria-label="Reset overlay position and zoom"
              >
                Reset
              </button>
              <button type="button" onClick={() => move(24, 0)} aria-label="Pan overlay right">
                →
              </button>
              <button type="button" onClick={() => move(0, 24)} aria-label="Pan overlay down">
                ↓
              </button>
            </fieldset>
          </div>
          <Text fontSize="xs" color="muted">
            The viewport stays in place when you switch detector maps. Indigo is a visual evidence
            ramp, not a verdict color.
          </Text>
        </VStack>
      </Card.Body>
    </Card.Root>
  );
}

export default function AnalysisResults({ results, originalUrl, detectors }: Props) {
  const verdict = verdictCopy[results.verdict];
  const rankedContributions = [...results.fusion.contributions].sort(
    (a, b) => Math.abs(b.signed_contribution) - Math.abs(a.signed_contribution),
  );
  return (
    <VStack align="stretch" gap={6} className="analysis-results">
      <Card.Root variant="outline" className="forensics-card" data-verdict={results.verdict}>
        <Card.Body>
          <VStack align="stretch" gap={4}>
            <Text className="eyebrow">Result / fused assessment</Text>
            <Box
              className={`verdict-panel verdict-panel--${verdict.tone}`}
              role="status"
              aria-live="polite"
            >
              <span className="verdict-icon" aria-hidden="true">
                {verdict.icon}
              </span>
              <Box>
                <Heading size="lg">{verdict.title}</Heading>
                <Text mt={1}>{verdict.description}</Text>
              </Box>
            </Box>
            <ScoreRuler
              score={results.score}
              threshold={SCORE_THRESHOLD}
              label="Fused evidence score"
            />
            <Text color="muted">{results.summary}</Text>
            <Text className="honesty-note" fontSize="sm">
              Evidence, not proof. Scores are estimated probabilities from the available signals,
              not proof of what happened to the image.
            </Text>
          </VStack>
        </Card.Body>
      </Card.Root>

      <Card.Root variant="outline" className="forensics-card">
        <Card.Body>
          <VStack align="stretch" gap={3}>
            <Box>
              <Text className="eyebrow">Signal profile</Text>
              <Heading size="md" mt={1}>
                Detector scores, with their limits
              </Heading>
              <Text color="muted" fontSize="sm" mt={1}>
                Magnitude is shown as a point, not a bar. The whisker is one Hanley–McNeil standard
                error when the service provides it.
              </Text>
            </Box>
            <Box className="chart-layout">
              <ScoreDotPlot results={results.detectors} detectors={detectors} />
              <DetectorTable results={results.detectors} detectors={detectors} />
            </Box>
          </VStack>
        </Card.Body>
      </Card.Root>

      <Card.Root variant="outline" className="forensics-card">
        <Card.Body>
          <VStack align="stretch" gap={3}>
            <Heading size="md">Why this result?</Heading>
            {rankedContributions.length ? (
              rankedContributions.map((contribution) => {
                const result = results.detectors.find((item) => item.id === contribution.id);
                const direction =
                  contribution.signed_contribution >= 0
                    ? 'toward manipulation'
                    : 'toward authentic';
                return (
                  <Box key={contribution.id} className="contribution-row">
                    <HStack justify="space-between" align="start" gap={4}>
                      <Box>
                        <Text fontWeight="bold">{detectorName(contribution.id, detectors)}</Text>
                        <Text fontSize="sm" color="muted">
                          {result?.reason ?? 'No detector explanation was returned.'}
                        </Text>
                      </Box>
                      <Text
                        className="contribution-value"
                        fontFamily="mono"
                        aria-label={`${contribution.signed_contribution.toFixed(2)} ${direction}`}
                      >
                        {contribution.signed_contribution >= 0 ? '+' : ''}
                        {contribution.signed_contribution.toFixed(2)}
                      </Text>
                    </HStack>
                  </Box>
                );
              })
            ) : (
              <Text color="muted">No fusion contributions were returned.</Text>
            )}
          </VStack>
        </Card.Body>
      </Card.Root>

      <OverlayViewer originalUrl={originalUrl} results={results} detectors={detectors} />

      <Box>
        <Text className="eyebrow">Detector details</Text>
        <Heading size="md" mt={1} mb={3}>
          Evidence returned by each method
        </Heading>
        <VStack align="stretch" gap={3}>
          {results.detectors.map((result) => (
            <EvidenceCard key={result.id} result={result} detectors={detectors} />
          ))}
        </VStack>
      </Box>

      {results.warnings.length > 0 && (
        <Card.Root variant="outline" className="forensics-card">
          <Card.Body>
            <Text fontWeight="bold">Caveats from the service</Text>
            <VStack align="stretch" gap={1} mt={2}>
              {results.warnings.map((warning) => (
                <Text key={warning} color="muted">
                  {warning}
                </Text>
              ))}
            </VStack>
          </Card.Body>
        </Card.Root>
      )}
    </VStack>
  );
}
