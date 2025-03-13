import { extendTheme } from '@chakra-ui/react'

// Custom color palette
const colors = {
  brand: {
    50: '#e6f6ff',
    100: '#b3e0ff',
    200: '#80cbff',
    300: '#4db5ff',
    400: '#1a9fff',
    500: '#0080e6',
    600: '#0066b3',
    700: '#004d80',
    800: '#00334d',
    900: '#001a26',
  },
}

// Custom component styles
const components = {
  Button: {
    baseStyle: {
      fontWeight: 'bold',
      borderRadius: 'md',
    },
    variants: {
      primary: {
        bg: 'brand.500',
        color: 'white',
        _hover: {
          bg: 'brand.600',
        },
      },
    },
  },
  Heading: {
    baseStyle: {
      fontWeight: '600',
      color: 'gray.800',
    },
  },
}

// Font configuration
const fonts = {
  heading: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  body: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
}

// Global styles
const styles = {
  global: {
    body: {
      bg: 'gray.50',
      color: 'gray.800',
    },
  },
}

// Create and export the theme
const theme = extendTheme({
  colors,
  components,
  fonts,
  styles,
})

export default theme
