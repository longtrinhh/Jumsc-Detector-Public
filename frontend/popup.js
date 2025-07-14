// Popup JavaScript for Jumpscare Detector
document.addEventListener('DOMContentLoaded', function() {
  // Initialize settings
  initializeSettings();
  

  
  // Update status
  updateStatus();
});

// Settings Management
function initializeSettings() {
  const sensitivitySlider = document.getElementById('sensitivity-slider');
  const sensitivityValue = document.getElementById('sensitivity-value');
  const autoCheck = document.getElementById('auto-check');
  const pauseVideo = document.getElementById('pause-video');
  
  // Load saved settings
  chrome.storage.sync.get({
    sensitivity: 0.4,
    autoCheck: true,
    pauseVideo: true
  }, function(items) {
    sensitivitySlider.value = items.sensitivity;
    sensitivityValue.textContent = items.sensitivity;
    autoCheck.checked = items.autoCheck;
    pauseVideo.checked = items.pauseVideo;
  });
  
  // Save sensitivity setting
  sensitivitySlider.addEventListener('input', function() {
    const value = this.value;
    sensitivityValue.textContent = value;
    chrome.storage.sync.set({ sensitivity: parseFloat(value) });
    
    // Send message to content script to update threshold
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      if (tabs[0] && tabs[0].url && tabs[0].url.includes('youtube.com')) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'updateThreshold',
          threshold: parseFloat(value)
        });
      }
    });
  });
  
  // Save auto-check setting
  autoCheck.addEventListener('change', function() {
    chrome.storage.sync.set({ autoCheck: this.checked });
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      if (tabs[0] && tabs[0].url && tabs[0].url.includes('youtube.com')) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'updateAutoCheck',
          autoCheck: this.checked
        });
      }
    });
  });
  
  // Save pause video setting
  pauseVideo.addEventListener('change', function() {
    chrome.storage.sync.set({ pauseVideo: this.checked });
  });
}



// Status Management
function updateStatus() {
  const statusText = document.getElementById('status-text');
  const statusDot = document.querySelector('.status-dot');
  
  // Check if we're on YouTube
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    if (tabs[0] && tabs[0].url && tabs[0].url.includes('youtube.com')) {
      statusText.textContent = 'Active on YouTube';
      statusDot.style.background = '#28a745';
    } else {
      statusText.textContent = 'Not on YouTube';
      statusDot.style.background = '#6c757d';
    }
  });
}



// Update status when popup opens
updateStatus(); 