// Configuration for the application
interface Config {
  apiBaseUrl: string;
  isProduction: boolean;
}

// Determine the environment
const isProduction = import.meta.env.PROD;

// Base configuration
const baseConfig: Config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000', // Default for development
  isProduction,
};

// Production overrides
if (isProduction) {
  // In production, use the environment variable or fall back to a default production URL
  baseConfig.apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'https://your-production-api-url.com';
  
  // Log the API URL in production for debugging (will be visible in browser console)
  console.log('Running in production mode with API URL:', baseConfig.apiBaseUrl);
} else {
  // In development, you can use the .env.development file
  baseConfig.apiBaseUrl = import.meta.env.VITE_API_BASE_URL || baseConfig.apiBaseUrl;
}

export default baseConfig;
