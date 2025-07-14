// Utility to get the YouTube video element (for pausing/resuming if needed)
function getYouTubeVideo() {
  return document.querySelector('video');
}

// Store the last checked URL to avoid duplicate checks
let lastCheckedUrl = null;
let isChecking = false;
let checkedUrls = new Set(); // Track all checked URLs in this session

// User settings
let userSettings = {
  sensitivity: 0.4,
  autoCheck: true,
  pauseVideo: true
};

// Load user settings
try {
  if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync) {
    chrome.storage.sync.get({
      sensitivity: 0.4,
      autoCheck: true,
      pauseVideo: true
    }, function(items) {
      userSettings = items;
    });
  }
} catch (error) {
  console.log('Could not load user settings:', error);
}

// Listen for messages from popup
try {
  if (typeof chrome !== 'undefined' && chrome.runtime) {
    chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
      if (request.action === 'updateThreshold') {
        userSettings.sensitivity = request.threshold;
      } else if (request.action === 'updateAutoCheck') {
        userSettings.autoCheck = request.autoCheck;
      }
    });
  }
} catch (error) {
  console.log('Could not set up message listener:', error);
}

function isYouTubeVideoOrShorts(url) {
  return (
    /^https?:\/\/(www\.)?youtube\.com\/watch\?v=/.test(url) ||
    /^https?:\/\/(www\.)?youtube\.com\/shorts\//.test(url)
  );
}

function normalizeYouTubeUrl(url) {
  // Extract just the video ID from YouTube URLs
  const watchMatch = url.match(/youtube\.com\/watch\?v=([^&]+)/);
  const shortsMatch = url.match(/youtube\.com\/shorts\/([^?&]+)/);
  
  if (watchMatch) {
    return `https://www.youtube.com/watch?v=${watchMatch[1]}`;
  } else if (shortsMatch) {
    return `https://www.youtube.com/shorts/${shortsMatch[1]}`;
  }
  
  return url; // Return original if no match
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Function to check for jumpscares automatically
async function autoCheckJumpscares() {
  const url = window.location.href;
  if (!isYouTubeVideoOrShorts(url)) return; // Only check for video/shorts
  if (!userSettings.autoCheck) return; // Auto-check disabled
  
  // Normalize URL to get just the video ID
  const normalizedUrl = normalizeYouTubeUrl(url);
  
  // Check if we've already analyzed this video in this session
  if (checkedUrls.has(normalizedUrl) || isChecking) {
    console.log('Video already checked:', normalizedUrl);
    console.log('Checked URLs:', Array.from(checkedUrls));
    return; // Already checked or already in progress
  }
  
  lastCheckedUrl = normalizedUrl;
  checkedUrls.add(normalizedUrl);
  isChecking = true;
  console.log('Starting analysis for:', normalizedUrl);
  console.log('Total checked URLs:', checkedUrls.size);

  // Optionally pause the video while checking
  const video = getYouTubeVideo();
  let wasPlaying = false;
  if (video && !video.paused && userSettings.pauseVideo) {
    wasPlaying = true;
    video.pause();
  }

  // Show small overlay in the bottom right
  let overlay = document.getElementById('jumpscare-auto-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'jumpscare-auto-overlay';
    overlay.style.position = 'fixed';
    overlay.style.bottom = '30px';
    overlay.style.right = '30px';
    overlay.style.background = 'rgba(30,30,30,0.95)';
    overlay.style.color = 'white';
    overlay.style.fontSize = '1.1em';
    overlay.style.padding = '18px 24px';
    overlay.style.borderRadius = '12px';
    overlay.style.boxShadow = '0 2px 12px rgba(0,0,0,0.3)';
    overlay.style.zIndex = '10000';
    overlay.style.display = 'flex';
    overlay.style.flexDirection = 'column';
    overlay.style.alignItems = 'flex-start';
    document.body.appendChild(overlay);
  } else {
    overlay.style.display = 'flex';
  }
  overlay.innerHTML = 'Checking for jumpscares...';

  try {
    // Get backend URL from configuration
    const backendUrl = window.JUMPSCARE_CONFIG ? window.JUMPSCARE_CONFIG.backendUrl : 'http://localhost:5001';
    console.log('Contacting backend:', backendUrl);
    
    const resp = await fetch(`${backendUrl}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        url: normalizedUrl, // Use normalized URL for backend
        threshold: userSettings.sensitivity
      })
    });
    const data = await resp.json();
    console.log('Backend response:', data);
    overlay.innerHTML = '';
    if (data.jumpscares && data.jumpscares.length > 0) {
      // Jumpscares detected: keep video paused, show list, require user action
      if (video && userSettings.pauseVideo) video.pause();
      
      const title = document.createElement('div');
      title.textContent = '⚠️ Jumpscares detected!';
      title.style.fontWeight = 'bold';
      title.style.marginBottom = '8px';
      overlay.appendChild(title);
      
      const list = document.createElement('ul');
      list.style.margin = '0 0 12px 0';
      list.style.paddingLeft = '18px';
      data.jumpscares.forEach(t => {
        const li = document.createElement('li');
        li.textContent = formatTime(t);
        list.appendChild(li);
      });
      overlay.appendChild(list);
      
      const buttonContainer = document.createElement('div');
      buttonContainer.style.display = 'flex';
      buttonContainer.style.gap = '8px';
      buttonContainer.style.marginTop = '8px';
      
      const playBtn = document.createElement('button');
      playBtn.textContent = 'Play Anyway';
      playBtn.style.padding = '6px 16px';
      playBtn.style.fontSize = '1em';
      playBtn.style.border = 'none';
      playBtn.style.borderRadius = '6px';
      playBtn.style.background = '#1976d2';
      playBtn.style.color = 'white';
      playBtn.style.cursor = 'pointer';
      playBtn.onclick = () => {
        overlay.style.display = 'none';
        if (video) video.play();
      };
      buttonContainer.appendChild(playBtn);
      

      
      overlay.appendChild(buttonContainer);
      

      
      // Keep video paused until user clicks
    } else if (data.jumpscares && data.jumpscares.length === 0) {
      // No jumpscares: show message and OK button
      const msg = document.createElement('div');
      msg.textContent = '✅ No jumpscares detected!';
      msg.style.marginBottom = '12px';
      overlay.appendChild(msg);
      
      const buttonContainer = document.createElement('div');
      buttonContainer.style.display = 'flex';
      buttonContainer.style.gap = '8px';
      buttonContainer.style.justifyContent = 'center';
      
      const okBtn = document.createElement('button');
      okBtn.textContent = 'OK';
      okBtn.style.padding = '6px 16px';
      okBtn.style.fontSize = '1em';
      okBtn.style.border = 'none';
      okBtn.style.borderRadius = '6px';
      okBtn.style.background = '#1976d2';
      okBtn.style.color = 'white';
      okBtn.style.cursor = 'pointer';
      okBtn.onclick = () => {
        overlay.style.display = 'none';
        if (wasPlaying && video) video.play();
      };
      buttonContainer.appendChild(okBtn);
      

      
      overlay.appendChild(buttonContainer);
      

    } else if (data.error) {
      overlay.innerHTML = 'Error: ' + data.error;
      const okBtn = document.createElement('button');
      okBtn.textContent = 'OK';
      okBtn.style.padding = '6px 16px';
      okBtn.style.fontSize = '1em';
      okBtn.style.border = 'none';
      okBtn.style.borderRadius = '6px';
      okBtn.style.background = '#1976d2';
      okBtn.style.color = 'white';
      okBtn.style.cursor = 'pointer';
      okBtn.onclick = () => {
        overlay.style.display = 'none';
        if (wasPlaying && video) video.play();
      };
      overlay.appendChild(okBtn);
    }
  } catch (e) {
    overlay.innerHTML = 'Failed to contact backend: ' + e.message;
    const okBtn = document.createElement('button');
    okBtn.textContent = 'OK';
    okBtn.style.padding = '6px 16px';
    okBtn.style.fontSize = '1em';
    okBtn.style.border = 'none';
    okBtn.style.borderRadius = '6px';
    okBtn.style.background = '#1976d2';
    okBtn.style.color = 'white';
    okBtn.style.cursor = 'pointer';
    okBtn.onclick = () => {
      overlay.style.display = 'none';
      if (wasPlaying && video) video.play();
    };
    overlay.appendChild(okBtn);
  } finally {
    isChecking = false;
  }
}

// Listen for URL changes (works for both regular and Shorts)
let lastUrl = window.location.href;
setInterval(() => {
  const currentUrl = window.location.href;
  if (currentUrl !== lastUrl) {
    lastUrl = currentUrl;
    console.log('URL changed:', currentUrl);
    autoCheckJumpscares();
  }
}, 1000);

// Clear cache on page load to allow re-checking if user refreshes
window.addEventListener('load', () => {
  console.log('Page loaded, clearing URL cache');
  checkedUrls.clear();
  lastCheckedUrl = null;
});

// Also clear cache when user navigates away and comes back
window.addEventListener('beforeunload', () => {
  console.log('Page unloading, clearing URL cache');
  checkedUrls.clear();
  lastCheckedUrl = null;
});

// Also run on initial load
autoCheckJumpscares();

 