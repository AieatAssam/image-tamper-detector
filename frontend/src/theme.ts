import { extendTheme } from '@chakra-ui/react';

export const theme = extendTheme({
  styles: {
    global: {
      body: {
        bg: 'gray.50',
      },
    },
  },
  components: {
    Button: {
      defaultProps: {
        colorScheme: 'blue',
      },
    },
    Card: {
      baseStyle: {
        container: {
          bg: 'white',
          boxShadow: 'lg',
          rounded: 'lg',
          p: 6,
        },
      },
    },
  },
}); 