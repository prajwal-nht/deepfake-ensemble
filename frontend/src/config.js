// Configuration for the application
export const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  // Add other configuration variables here
};

// Make sure to add this line for TypeScript support
export default config;
