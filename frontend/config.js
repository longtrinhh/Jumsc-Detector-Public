// Backend Configuration for Jumpscare Detector Extension

const CONFIG = {
  // Production Configuration (HTTPS required for YouTube)
  PRODUCTION: {
    BACKEND_URL: 'https://random.random',  // Replace with your HTTPS domain
    DESCRIPTION: 'Production HTTPS backend'
  },
  
  // Development Configuration (Local testing)
  DEVELOPMENT: {
    BACKEND_URL: 'http://localhost:5001',
    DESCRIPTION: 'Local development backend'
  },
  
  // Ngrok Configuration (Quick HTTPS testing)
  NGROK: {
    BACKEND_URL: 'https://abc123.ngrok.io',  // Replace with your ngrok URL
    DESCRIPTION: 'Ngrok tunnel for testing'
  },
  
  // VPS Direct (if you set up HTTPS on VPS)
  VPS_HTTPS: {
    BACKEND_URL: 'https://192.168.1.100',  // Your VPS with SSL
    DESCRIPTION: 'VPS with SSL certificate'
  }
};

// Current Configuration - Change this to switch backends
const CURRENT_ENV = 'PRODUCTION';  // Options: 'PRODUCTION', 'DEVELOPMENT', 'NGROK', 'VPS_HTTPS'

// Get the current backend URL
function getBackendUrl() {
  const config = CONFIG[CURRENT_ENV];
  console.log(`Using ${config.DESCRIPTION}: ${config.BACKEND_URL}`);
  return config.BACKEND_URL;
}

// Export for use in content.js
window.JUMPSCARE_CONFIG = {
  backendUrl: getBackendUrl(),
  environment: CURRENT_ENV
}; 