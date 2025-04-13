import {
  Badge,
  Card,
  Heading,
  Image,
  Progress,
  Text,
  VStack,
} from '@chakra-ui/react';
import { AnalysisResponse, CombinedAnalysisResponse } from '../types/api';

interface Props {
  results: AnalysisResponse | CombinedAnalysisResponse;
}

function SingleAnalysisResult({ result }: { result: AnalysisResponse }) {
  return (
    <VStack spacing={4} align="stretch">
      <Heading size="md">{result.analysis_type} Analysis Results</Heading>
      <Badge
        colorScheme={result.is_tampered ? 'red' : 'green'}
        fontSize="md"
        p={2}
        borderRadius="md"
      >
        {result.is_tampered ? 'Potential Tampering Detected' : 'No Tampering Detected'}
      </Badge>
      <Text>
        Confidence Score:{' '}
        <Progress
          value={result.confidence_score * 100}
          colorScheme={result.is_tampered ? 'red' : 'green'}
          hasStripe
        />
      </Text>
      <Image
        src={`data:image/png;base64,${result.visualization_base64}`}
        alt={`${result.analysis_type} visualization`}
        maxH="300px"
        objectFit="contain"
      />
      <VStack align="stretch" spacing={2}>
        <Text fontWeight="bold">Analysis Details:</Text>
        <Text>{result.details.description}</Text>
        {result.details.edge_discontinuity !== undefined && (
          <Text>Edge Discontinuity: {result.details.edge_discontinuity.toFixed(2)}</Text>
        )}
        {result.details.texture_variance !== undefined && (
          <Text>Texture Variance: {result.details.texture_variance.toFixed(2)}</Text>
        )}
        {result.details.noise_consistency !== undefined && (
          <Text>Noise Consistency: {result.details.noise_consistency.toFixed(2)}</Text>
        )}
        {result.details.compression_artifacts !== undefined && (
          <Text>Compression Artifacts: {result.details.compression_artifacts.toFixed(2)}</Text>
        )}
        {result.details.matching_proportion !== undefined && (
          <Text>Matching Proportion: {result.details.matching_proportion.toFixed(2)}</Text>
        )}
      </VStack>
    </VStack>
  );
}

export default function AnalysisResults({ results }: Props) {
  if ('analysis_type' in results) {
    return (
      <Card>
        <SingleAnalysisResult result={results as AnalysisResponse} />
      </Card>
    );
  }

  const combinedResults = results as CombinedAnalysisResponse;
  return (
    <VStack spacing={6} align="stretch">
      <Card>
        <VStack spacing={4} align="stretch">
          <Heading size="lg">Combined Analysis Results</Heading>
          <Badge
            colorScheme={combinedResults.is_tampered ? 'red' : 'green'}
            fontSize="lg"
            p={2}
            borderRadius="md"
          >
            {combinedResults.is_tampered
              ? 'Potential Tampering Detected'
              : 'No Tampering Detected'}
          </Badge>
          <Text>
            Overall Confidence Score:{' '}
            <Progress
              value={combinedResults.confidence_score * 100}
              colorScheme={combinedResults.is_tampered ? 'red' : 'green'}
              hasStripe
              size="lg"
            />
          </Text>
        </VStack>
      </Card>

      {combinedResults.ela_result && (
        <Card>
          <SingleAnalysisResult result={combinedResults.ela_result} />
        </Card>
      )}

      {combinedResults.prnu_result && (
        <Card>
          <SingleAnalysisResult result={combinedResults.prnu_result} />
        </Card>
      )}

      {combinedResults.entropy_result && (
        <Card>
          <SingleAnalysisResult result={combinedResults.entropy_result} />
        </Card>
      )}
    </VStack>
  );
} 