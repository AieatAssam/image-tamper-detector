import { Box, Button, Card, HStack, Image, Text, VStack } from '@chakra-ui/react';
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
            <Box
              {...getRootProps()}
              role="button"
              tabIndex={0}
              borderWidth="2px"
              borderStyle="dashed"
              borderColor={isDragActive ? 'signal' : 'line'}
              borderRadius="md"
              p={{ base: 8, md: 12 }}
              textAlign="center"
              cursor="pointer"
            >
              <input {...getInputProps()} aria-label="Choose image file" />
              <Text fontWeight="bold">
                {isDragActive ? 'Drop the image here' : 'Drop an image here or choose a file'}
              </Text>
              <Text color="muted" fontSize="sm" mt={2}>
                JPEG, PNG, WebP, or TIFF. Maximum 12 MB.
              </Text>
            </Box>
            {error && (
              <Text role="alert" color="signal">
                {error}
              </Text>
            )}
            {file && (
              <HStack justify="space-between" align="center" wrap="wrap" gap={3}>
                <Text fontFamily="mono" fontSize="sm">
                  {file.name} · {Math.round(file.size / 1024)} KB
                </Text>
                <HStack>
                  <Button onClick={() => void runAnalysis()} loading={loading} disabled={loading}>
                    Analyze image
                  </Button>
                  {loading && (
                    <Button variant="outline" onClick={() => abortRef.current?.abort()}>
                      Cancel
                    </Button>
                  )}
                </HStack>
              </HStack>
            )}
            {loading && (
              <Text aria-live="polite" color="muted">
                Uploading and analyzing{progress ? ` · ${progress}% uploaded` : '…'}
              </Text>
            )}
          </VStack>
        </Card.Body>
      </Card.Root>

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
