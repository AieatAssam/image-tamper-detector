import { Badge, Box, Card, Heading, HStack, Image, Progress, Text, VStack } from '@chakra-ui/react';
import { useState } from 'react';
import type { AnalysisResponse, DetectorInfo, DetectorResult, Verdict } from '../types/api';

interface Props {
  results: AnalysisResponse;
  originalUrl: string;
  detectors: DetectorInfo[];
}

const verdictCopy: Record<
  Verdict,
  { title: string; description: string; color: string; icon: string }
> = {
  manipulated: {
    title: 'Strong manipulation signal',
    description: 'Several available signals point toward a manipulated image.',
    color: 'signal',
    icon: '▲',
  },
  likely_manipulated: {
    title: 'Likely manipulated',
    description: 'The evidence leans toward image manipulation, but it is not proof.',
    color: 'signal',
    icon: '↗',
  },
  inconclusive: {
    title: 'Inconclusive evidence',
    description: 'There is not enough applicable evidence for a reliable direction.',
    color: 'uncertain',
    icon: '?',
  },
  likely_authentic: {
    title: 'Likely authentic',
    description: 'The available evidence leans toward an authentic image.',
    color: 'trustworthy',
    icon: '✓',
  },
  authentic: {
    title: 'Authentic signal',
    description: 'The available signals support an authentic image.',
    color: 'trustworthy',
    icon: '✓',
  },
};

function detectorName(id: string, detectors: DetectorInfo[]): string {
  return detectors.find((detector) => detector.id === id)?.name ?? id.replaceAll('_', ' ');
}

function scoreText(score: number | null): string {
  return score === null ? 'No score' : `${Math.round(score * 100)}%`;
}

function ScoreBar({
  score,
  threshold,
  label,
  color,
}: {
  score: number;
  threshold: number;
  label: string;
  color: string;
}) {
  return (
    <Box>
      <HStack justify="space-between" mb={1}>
        <Text fontSize="sm" color="muted">
          {label}
        </Text>
        <Text fontFamily="mono" fontSize="sm" fontWeight="bold">
          {scoreText(score)}
        </Text>
      </HStack>
      <Progress.Root value={score * 100} colorPalette={color} size="sm" aria-label={label}>
        <Progress.Track>
          <Progress.Range />
        </Progress.Track>
      </Progress.Root>
      <Text fontSize="xs" color="muted" mt={1}>
        Estimated probability this image was manipulated. Detector threshold:{' '}
        {Math.round(threshold * 100)}%.
      </Text>
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
  const stateColor =
    result.state === 'applicable' ? (result.flagged ? 'signal' : 'trustworthy') : 'uncertain';
  return (
    <Card.Root variant="outline" data-state={result.state}>
      <Card.Body>
        <details>
          <summary>
            <HStack display="inline-flex" gap={3} ml={2}>
              <Text fontWeight="bold">{info?.name ?? result.id}</Text>
              <Badge colorPalette={stateColor}>
                {result.state === 'not_applicable' ? 'Could not assess' : result.state}
              </Badge>
            </HStack>
          </summary>
          <VStack align="stretch" gap={4} mt={4}>
            {info?.description && <Text color="muted">{info.description}</Text>}
            <Text>{result.error ?? result.reason}</Text>
            {result.score !== null && (
              <ScoreBar
                score={result.score}
                threshold={result.threshold}
                label={`${info?.name ?? result.id} score`}
                color={stateColor}
              />
            )}
            {Object.keys(result.metrics).length > 0 && (
              <Box>
                <Text fontWeight="bold" mb={2}>
                  Measurements
                </Text>
                <VStack align="stretch" gap={1} fontFamily="mono" fontSize="sm">
                  {Object.entries(result.metrics).map(([key, value]) => (
                    <HStack key={key} justify="space-between">
                      <Text color="muted">{key.replaceAll('_', ' ')}</Text>
                      <Text>{value.toFixed(3)}</Text>
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

function OverlayViewer({ originalUrl, results, detectors }: Props) {
  const mapped = results.detectors.filter((result) => result.visualization_png_base64);
  const [selectedId, setSelectedId] = useState(mapped[0]?.id ?? '');
  const [opacity, setOpacity] = useState(65);
  const [divider, setDivider] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const selected = mapped.find((result) => result.id === selectedId) ?? mapped[0];
  if (!selected?.visualization_png_base64) return null;

  const move = (x: number, y: number) => setPan((value) => ({ x: value.x + x, y: value.y + y }));
  const transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;
  return (
    <Card.Root variant="outline">
      <Card.Body>
        <VStack align="stretch" gap={4}>
          <Box>
            <Text
              fontFamily="mono"
              fontSize="xs"
              color="signal"
              textTransform="uppercase"
              letterSpacing="0.12em"
            >
              Compare / overlay
            </Text>
            <Heading size="md" mt={1}>
              Inspect the visual evidence
            </Heading>
            <Text color="muted" fontSize="sm" mt={1}>
              Original underneath. The detector map is clipped by the swipe divider and blended at
              your chosen opacity.
            </Text>
          </Box>
          <label>
            <Text fontSize="sm" fontWeight="bold" mb={1}>
              Evidence layer
            </Text>
            <select
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
            position="relative"
            overflow="hidden"
            minH={{ base: '240px', md: '420px' }}
            bg="black"
            borderRadius="md"
            role="img"
            aria-label="Original image with detector overlay"
          >
            <Image
              src={originalUrl}
              alt="Original uploaded image"
              position="absolute"
              inset={0}
              w="100%"
              h="100%"
              objectFit="contain"
              transform={transform}
              transition="transform 0.15s"
            />
            <Box
              position="absolute"
              inset={0}
              clipPath={`inset(0 ${100 - divider}% 0 0)`}
              opacity={opacity / 100}
            >
              <Image
                src={`data:image/png;base64,${selected.visualization_png_base64}`}
                alt={`${detectorName(selected.id, detectors)} heatmap`}
                w="100%"
                h="100%"
                objectFit="contain"
                transform={transform}
                transition="transform 0.15s"
              />
            </Box>
            <Box
              position="absolute"
              top={0}
              bottom={0}
              left={`${divider}%`}
              w="2px"
              bg="white"
              boxShadow="0 0 0 1px rgba(0,0,0,.35)"
              aria-hidden="true"
            />
          </Box>
          <VStack align="stretch" gap={3}>
            <label>
              <HStack justify="space-between">
                <Text fontSize="sm">Heatmap opacity</Text>
                <Text fontFamily="mono" fontSize="sm">
                  {opacity}%
                </Text>
              </HStack>
              <input
                type="range"
                min="0"
                max="100"
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
                aria-label="Heatmap opacity"
                style={{ width: '100%' }}
              />
            </label>
            <label>
              <HStack justify="space-between">
                <Text fontSize="sm">Swipe divider</Text>
                <Text fontFamily="mono" fontSize="sm">
                  {divider}%
                </Text>
              </HStack>
              <input
                type="range"
                min="0"
                max="100"
                value={divider}
                onChange={(event) => setDivider(Number(event.target.value))}
                aria-label="Swipe divider"
                style={{ width: '100%' }}
              />
            </label>
            <label>
              <HStack justify="space-between">
                <Text fontSize="sm">Zoom</Text>
                <Text fontFamily="mono" fontSize="sm">
                  {zoom.toFixed(1)}×
                </Text>
              </HStack>
              <input
                type="range"
                min="1"
                max="3"
                step="0.1"
                value={zoom}
                onChange={(event) => setZoom(Number(event.target.value))}
                aria-label="Overlay zoom"
                style={{ width: '100%' }}
              />
            </label>
            <HStack gap={2} wrap="wrap">
              <Text fontSize="sm" color="muted">
                Pan
              </Text>
              <button type="button" onClick={() => move(0, -20)} aria-label="Pan overlay up">
                ↑
              </button>
              <button type="button" onClick={() => move(-20, 0)} aria-label="Pan overlay left">
                ←
              </button>
              <button
                type="button"
                onClick={() => setPan({ x: 0, y: 0 })}
                aria-label="Reset overlay position"
              >
                Reset
              </button>
              <button type="button" onClick={() => move(20, 0)} aria-label="Pan overlay right">
                →
              </button>
              <button type="button" onClick={() => move(0, 20)} aria-label="Pan overlay down">
                ↓
              </button>
            </HStack>
          </VStack>
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
    <VStack align="stretch" gap={6}>
      <Card.Root variant="outline" data-verdict={results.verdict}>
        <Card.Body>
          <VStack align="stretch" gap={4}>
            <Text
              fontFamily="mono"
              fontSize="xs"
              color="signal"
              textTransform="uppercase"
              letterSpacing="0.12em"
            >
              Result / fused assessment
            </Text>
            <Box
              role="status"
              aria-live="polite"
              borderLeftWidth="5px"
              borderColor={verdict.color}
              pl={4}
            >
              <HStack gap={3} align="start">
                <Text fontSize="2xl" fontWeight="bold" aria-hidden="true">
                  {verdict.icon}
                </Text>
                <Box>
                  <Heading size="lg">{verdict.title}</Heading>
                  <Text mt={1}>{verdict.description}</Text>
                </Box>
              </HStack>
            </Box>
            <ScoreBar
              score={results.score}
              threshold={0.5}
              label="Fused score"
              color={verdict.color}
            />
            <Text color="muted">{results.summary}</Text>
            <Text fontSize="sm" color="muted">
              Scores are estimated probabilities from the available evidence, not proof of what
              happened to the image.
            </Text>
          </VStack>
        </Card.Body>
      </Card.Root>

      <Card.Root variant="outline">
        <Card.Body>
          <VStack align="stretch" gap={3}>
            <Heading size="md">Why this result?</Heading>
            {rankedContributions.length ? (
              rankedContributions.map((contribution) => {
                const result = results.detectors.find((item) => item.id === contribution.id);
                return (
                  <Box key={contribution.id} borderTopWidth="1px" borderColor="line" pt={3}>
                    <HStack justify="space-between" align="start" gap={4}>
                      <Box>
                        <Text fontWeight="bold">{detectorName(contribution.id, detectors)}</Text>
                        <Text fontSize="sm" color="muted">
                          {result?.reason ?? 'No detector explanation was returned.'}
                        </Text>
                      </Box>
                      <Text
                        fontFamily="mono"
                        color={contribution.signed_contribution >= 0 ? 'signal' : 'trustworthy'}
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
        <Heading size="md" mb={3}>
          Detector evidence
        </Heading>
        <VStack align="stretch" gap={3}>
          {results.detectors.map((result) => (
            <EvidenceCard key={result.id} result={result} detectors={detectors} />
          ))}
        </VStack>
      </Box>

      {results.warnings.length > 0 && (
        <Card.Root variant="outline">
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
