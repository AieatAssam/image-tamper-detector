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
      <a className="skip-link" href="#main-content">
        Skip to analysis
      </a>
      <Container maxW="1200px" py={{ base: 5, md: 10 }}>
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
                Inspect the evidence
              </Heading>
              <Text color="muted" mt={2} maxW="600px">
                Upload one image to compare forensic signals, see which checks could run, and
                understand what each result can and cannot tell you.
              </Text>
            </Box>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsDark((value) => !value)}
              aria-label={isDark ? 'Use light colour theme' : 'Use dark colour theme'}
              aria-pressed={isDark}
            >
              {isDark ? 'Use light theme' : 'Use dark theme'}
            </Button>
          </HStack>
          <Box as="main" id="main-content">
            <ImageAnalyzer />
          </Box>
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App;
