import { Box, Button, Card, Heading, HStack, Image, Text, VStack } from '@chakra-ui/react';
import { useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  AnalysisError,
  analyze,
  errorMessage,
  getDetectors,
  validateImageDimensions,
} from '../api/client';
import type { AnalysisResponse, DetectorInfo } from '../types/api';
import AnalysisResults from './AnalysisResults';

function DetectorGuide({ detectors }: { detectors: DetectorInfo[] }) {
  if (!detectors.length) return null;

  return (
    <Card.Root variant="outline" className="forensics-card detector-guide">
      <Card.Body>
        <VStack align="stretch" gap={4}>
          <Box>
            <Text className="eyebrow">Before you upload</Text>
            <Heading size="md" mt={1}>
              What these checks look for
            </Heading>
            <Text color="muted" fontSize="sm" mt={1}>
              Each detector checks a different kind of image trace. A detector can be useful,
              inconclusive, or outside its scope; none proves who made or edited an image.
            </Text>
          </Box>
          <VStack align="stretch" gap={0} className="detector-guide-list">
            {detectors.map((detector) => (
              <Box key={detector.id} className="detector-guide-row">
                <HStack align="start" justify="space-between" gap={4}>
                  <Box minW={0}>
                    <Text fontWeight="bold">{detector.name}</Text>
                    <Text color="muted" fontSize="sm" mt={1}>
                      {detector.description}
                    </Text>
                  </Box>
                  <Text className="detector-family" flex="0 0 auto">
                    {detector.family}
                  </Text>
                </HStack>
                <Text className="detector-scope" fontSize="sm" mt={2}>
                  Works on {detector.applicable_formats.join(', ')}
                  {detector.produces_map ? ' · produces a visual map' : ''}
                </Text>
                {detector.limitations.length > 0 && (
                  <Text color="muted" fontSize="sm" mt={1}>
                    Limit: {detector.limitations.join(' ')}
                  </Text>
                )}
              </Box>
            ))}
          </VStack>
        </VStack>
      </Card.Body>
    </Card.Root>
  );
}

export default function ImageAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [detectors, setDetectors] = useState<DetectorInfo[]>([]);
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [history, setHistory] = useState<AnalysisResponse[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void getDetectors(controller.signal)
      .then((response) => setDetectors(response.detectors))
      .catch((reason: unknown) => {
        if (!(reason instanceof AnalysisError) || reason.kind !== 'aborted')
          setError(errorMessage(reason));
      });
    return () => {
      controller.abort();
      abortRef.current?.abort();
    };
  }, []);

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  const chooseFile = async (nextFile: File) => {
    setError('');
    try {
      await validateImageDimensions(nextFile);
      setPreviewUrl((oldUrl) => {
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        return URL.createObjectURL(nextFile);
      });
      setFile(nextFile);
      setResults(null);
      setProgress(0);
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
      'image/tiff': ['.tif', '.tiff'],
    },
    maxFiles: 1,
    multiple: false,
    onDrop: (acceptedFiles) => {
      const nextFile = acceptedFiles[0];
      if (nextFile) void chooseFile(nextFile);
    },
  });

  const runAnalysis = async () => {
    if (!file || loading) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError('');
    setProgress(0);
    try {
      const response = await analyze(file, {
        includeMaps: true,
        signal: controller.signal,
        onUploadProgress: setProgress,
      });
      setResults(response);
      setHistory((previous) =>
        [response, ...previous.filter((item) => item.image.sha256 !== response.image.sha256)].slice(
          0,
          5,
        ),
      );
    } catch (reason) {
      if (!(reason instanceof AnalysisError) || reason.kind !== 'aborted')
        setError(errorMessage(reason));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setLoading(false);
    }
  };

  return (
    <VStack align="stretch" gap={6}>
      <Card.Root variant="outline">
        <Card.Body>
          <VStack align="stretch" gap={4}>
            <Box>
              <Text className="eyebrow">Step 1 / choose an image</Text>
              <Heading size="md" mt={1}>
                Start with the original file
              </Heading>
              <Text color="muted" fontSize="sm" mt={1}>
                The file stays in this analysis session. We inspect the pixels and available
                metadata, then show the evidence behind the result.
              </Text>
            </Box>
            <Box
              {...getRootProps()}
              className="upload-dropzone"
              role="button"
              tabIndex={0}
              aria-label="Choose an image file to analyze"
              borderWidth="2px"
              borderStyle="dashed"
              borderColor={isDragActive ? 'signal' : 'line'}
              borderRadius="md"
              p={{ base: 8, md: 12 }}
              textAlign="center"
              cursor="pointer"
            >
              <input {...getInputProps()} aria-label="Choose image file" />
              <Text fontWeight="bold" fontSize="lg">
                {isDragActive ? 'Drop the image here' : 'Drop an image here or choose a file'}
              </Text>
              <Text color="muted" fontSize="sm" mt={2}>
                JPEG, PNG, WebP, or TIFF · maximum 12 MB · one file at a time
              </Text>
            </Box>
            {error && (
              <Box className="error-panel" role="alert">
                <Text fontWeight="bold">We could not analyze that image</Text>
                <Text mt={1}>{error}</Text>
              </Box>
            )}
            {file && (
              <Box className="selected-file">
                <Box minW={0}>
                  <Text fontSize="sm" color="muted">
                    Selected file
                  </Text>
                  <Text fontFamily="mono" fontSize="sm" overflowWrap="anywhere">
                    {file.name}
                  </Text>
                  <Text fontSize="sm" color="muted" mt={1}>
                    {Math.round(file.size / 1024)} KB · ready to inspect
                  </Text>
                </Box>
                <HStack className="upload-actions">
                  <Button onClick={() => void runAnalysis()} loading={loading} disabled={loading}>
                    Analyze image
                  </Button>
                  {loading && (
                    <Button variant="outline" onClick={() => abortRef.current?.abort()}>
                      Cancel
                    </Button>
                  )}
                </HStack>
              </Box>
            )}
            {loading && (
              <Box className="progress-panel" aria-live="polite">
                <HStack justify="space-between" gap={3}>
                  <Text>Uploading and analyzing</Text>
                  <Text fontFamily="mono" fontSize="sm">
                    {progress ? `${progress}% uploaded` : 'Preparing'}
                  </Text>
                </HStack>
                <progress value={progress} max="100" aria-label="Analysis upload progress" />
                <Text color="muted" fontSize="sm" mt={2}>
                  This can take a little while because several independent checks run together.
                </Text>
              </Box>
            )}
          </VStack>
        </Card.Body>
      </Card.Root>

      <DetectorGuide detectors={detectors} />

      {previewUrl && (
        <Image
          src={previewUrl}
          alt="Selected image preview"
          maxH="420px"
          w="100%"
          objectFit="contain"
          borderRadius="md"
        />
      )}
      {results && previewUrl && (
        <AnalysisResults results={results} originalUrl={previewUrl} detectors={detectors} />
      )}
      {history.length > 0 && (
        <Text color="muted" fontSize="sm" aria-label="Analysis history">
          This session has {history.length} saved result{history.length === 1 ? '' : 's'}.
        </Text>
      )}
      <Text fontSize="xs" color="muted">
        This tool provides probabilistic forensic signals, not proof of an image’s origin or editing
        history.
      </Text>
    </VStack>
  );
}
