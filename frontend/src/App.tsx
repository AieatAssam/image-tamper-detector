import {
  Box,
  Button,
  ChakraProvider,
  Container,
  Heading,
  HStack,
  Text,
  VStack,
} from '@chakra-ui/react';
import { useEffect, useState } from 'react';
import ImageAnalyzer from './components/ImageAnalyzer';
import { system } from './theme';

function App() {
  const [isDark, setIsDark] = useState(
    () =>
      typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches,
  );

  useEffect(() => {
    document.documentElement.dataset.theme = isDark ? 'dark' : 'light';
  }, [isDark]);

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (event: MediaQueryListEvent) => setIsDark(event.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  return (
    <ChakraProvider value={system}>
      <Container maxW="1100px" py={{ base: 6, md: 12 }}>
        <VStack gap={{ base: 6, md: 10 }} align="stretch">
          <HStack justify="space-between" align="start" gap={4}>
            <Box>
              <Text
                fontFamily="mono"
                fontSize="xs"
                letterSpacing="0.18em"
                color="signal"
                textTransform="uppercase"
              >
                Evidence desk / 01
              </Text>
              <Heading as="h1" size={{ base: 'xl', md: '2xl' }} mt={2} letterSpacing="-0.04em">
                Is this image real?
              </Heading>
              <Text color="muted" mt={2} maxW="600px">
                Upload an image and inspect the signals that support or challenge its provenance.
              </Text>
            </Box>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsDark((value) => !value)}
              aria-label={isDark ? 'Use light colour theme' : 'Use dark colour theme'}
            >
              {isDark ? 'Light mode' : 'Dark mode'}
            </Button>
          </HStack>
          <ImageAnalyzer />
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App;
