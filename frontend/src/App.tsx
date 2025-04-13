import { ChakraProvider, Container, Heading, VStack } from '@chakra-ui/react';
import ImageAnalyzer from './components/ImageAnalyzer';
import { theme } from './theme';

function App() {
  return (
    <ChakraProvider theme={theme}>
      <Container maxW="container.xl" py={8}>
        <VStack spacing={8} align="stretch">
          <Heading as="h1" size="2xl" textAlign="center" color="blue.600">
            Image Tampering Detection
          </Heading>
          <ImageAnalyzer />
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App; 