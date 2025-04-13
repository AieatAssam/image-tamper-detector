import { useState } from 'react';
import {
  Box,
  Button,
  Card,
  Grid,
  HStack,
  Image,
  Text,
  useToast,
  VStack,
} from '@chakra-ui/react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { AnalysisResponse, CombinedAnalysisResponse } from '../types/api';
import AnalysisResults from './AnalysisResults';

type AnalysisType = 'ela' | 'prnu' | 'entropy' | 'combined';

export default function ImageAnalyzer() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResponse | CombinedAnalysisResponse | null>(null);
  const toast = useToast();

  const onDrop = (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
    },
    maxFiles: 1,
  });

  const analyzeImage = async (type: AnalysisType) => {
    if (!selectedImage) return;

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      setIsAnalyzing(true);
      const response = await axios.post(`/analyze/${type}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data);
    } catch (error) {
      toast({
        title: 'Analysis failed',
        description: 'There was an error analyzing the image. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <VStack spacing={6} align="stretch">
      <Card>
        <Box
          {...getRootProps()}
          p={6}
          border="2px dashed"
          borderColor={isDragActive ? 'blue.500' : 'gray.200'}
          borderRadius="lg"
          textAlign="center"
          cursor="pointer"
          transition="all 0.2s"
          _hover={{ borderColor: 'blue.500' }}
        >
          <input {...getInputProps()} />
          <Text>
            {isDragActive
              ? 'Drop the image here...'
              : 'Drag and drop an image here, or click to select'}
          </Text>
        </Box>
      </Card>

      {imagePreview && (
        <Grid templateColumns={{ base: '1fr', md: '1fr 1fr' }} gap={6}>
          <Card>
            <VStack spacing={4}>
              <Image
                src={imagePreview}
                alt="Selected image"
                maxH="400px"
                objectFit="contain"
              />
              <HStack spacing={4} wrap="wrap" justify="center">
                <Button
                  onClick={() => analyzeImage('ela')}
                  isLoading={isAnalyzing}
                  loadingText="Analyzing"
                >
                  ELA Analysis
                </Button>
                <Button
                  onClick={() => analyzeImage('prnu')}
                  isLoading={isAnalyzing}
                  loadingText="Analyzing"
                >
                  PRNU Analysis
                </Button>
                <Button
                  onClick={() => analyzeImage('entropy')}
                  isLoading={isAnalyzing}
                  loadingText="Analyzing"
                >
                  Entropy Analysis
                </Button>
                <Button
                  onClick={() => analyzeImage('combined')}
                  isLoading={isAnalyzing}
                  loadingText="Analyzing"
                  colorScheme="green"
                >
                  Combined Analysis
                </Button>
              </HStack>
            </VStack>
          </Card>

          {results && <AnalysisResults results={results} />}
        </Grid>
      )}
    </VStack>
  );
} 