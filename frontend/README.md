# Frontend - Browser Extension

**Browser extension that automatically detects and warns about jumpscares in YouTube videos.**

## Quick Start

### **1. Install Extension**
1. Open Chrome/Edge browser
2. Go to `chrome://extensions/`
3. Enable **"Developer mode"** (top-right toggle)
4. Click **"Load unpacked"**
5. Select this `frontend/` folder
6. Extension icon should appear in toolbar!

### **2. Configure Backend URL**
Update the backend URL in `content.js`:

```javascript
// Line ~63 in content.js
const resp = await fetch('http://YOUR_VPS_IP:5001/analyze', {
```

**Replace with:**
```javascript
// Local backend
const resp = await fetch('http://localhost:5001/analyze', {

// VPS backend  
const resp = await fetch('http://123.456.789.012:5001/analyze', {

// Domain backend
const resp = await fetch('https://api.yourdomain.com/analyze', {
```

### **3. Test on YouTube**
1. Go to any YouTube video
2. Extension should automatically analyze it
3. Look for warning overlays if jumpscares detected! ⚠️

---

## Files Overview

### **Core Extension**
- `manifest.json` - **Extension configuration** (permissions, scripts, icons)
- `content.js` - **Main YouTube integration** (auto-detection, UI)
- `background.js` - Service worker (minimal functionality)

### **🎨 User Interface**
- `popup.html` - Extension popup when clicking icon
- `styles.css` - Popup styling
- `icon*.png` - Extension icons (16px, 19px, 38px, 48px, 128px)

---

## 🔄 How It Works

1. **Content script loads** on YouTube pages
2. **Detects video URL changes** (including YouTube Shorts)
3. **Pauses video** and sends URL to backend API
4. **Backend analyzes audio** using ML models
5. **Displays warning overlay** with jumpscare timestamps
6. **User chooses** to continue or skip video

---

## Features

### **Automatic Detection**
- Monitors YouTube URL changes
- Works on regular videos AND YouTube Shorts
- No manual activation required

### **Smart UI**
- **Video paused** during analysis for safety
- **Clear warnings** with jumpscare timestamps  
- **User choice** - "Play Anyway" or skip
- **Non-intrusive** overlay design

### **Performance**
- **Caching** - Faster results for analyzed videos
- **Error handling** - Graceful fallbacks
- **Memory efficient** - Minimal resource usage

---

## 🛠 Development

### **Local Development**
1. **Make changes** to any file in `frontend/`
2. **Go to** `chrome://extensions/`
3. **Click refresh** button on "Jumpscare Detector" extension
4. **Test changes** on YouTube

### **Key Files to Modify**

**`content.js` - Main functionality:**
```javascript
// Backend URL configuration
const resp = await fetch('http://localhost:5001/analyze', {

// UI styling
overlay.style.background = 'rgba(30,30,30,0.95)';

// Detection threshold
body: JSON.stringify({ url, threshold: 0.4 })
```

**`manifest.json` - Extension config:**
```json
{
  "name": "Jumpscare Detector for YouTube",
  "version": "1.0.0",
  "permissions": ["activeTab", "scripting"],
  "host_permissions": ["https://www.youtube.com/*"]
}
```

**`popup.html` - Extension popup:**
```html
<h1>Jumpscare Detector</h1>
<p>Protecting you from unexpected scares!</p>
```

### **Testing Workflow**
1. **Local backend**: Start `backend/backend_ml.py`
2. **Load extension**: Point to this `frontend/` folder
3. **Test videos**: Try known jumpscare videos
4. **Check console**: Look for errors in browser DevTools

---

## 🎨 Customization

### **UI Styling**
Modify the overlay appearance in `content.js`:

```javascript
// Position
overlay.style.bottom = '30px';
overlay.style.right = '30px';

// Colors
overlay.style.background = 'rgba(30,30,30,0.95)';
overlay.style.color = 'white';

// Size
overlay.style.fontSize = '1.1em';
overlay.style.padding = '18px 24px';
```

### **Detection Sensitivity**
Adjust the threshold for more/less sensitive detection:

```javascript
// In content.js, around line 65
body: JSON.stringify({ 
  url,
  threshold: 0.3  // Lower = more sensitive
})
```

### **Extension Info**
Update extension details in `manifest.json`:

```json
{
  "name": "Your Custom Jumpscare Detector",
  "description": "Your custom description",
  "version": "1.0.0"
}
```

---

## Production Deployment

### **Chrome Web Store**
1. **Package extension**:
   ```bash
   zip -r jumpscare-detector.zip frontend/
   ```

2. **Update manifest**:
   ```json
   {
     "version": "1.0.0",
     "description": "Professional description for store"
   }
   ```

3. **Submit to Chrome Web Store**:
   - Pay $5 developer fee
   - Upload packaged extension
   - Fill store listing details
   - Wait for review (~3-7 days)

### **Firefox Add-ons**
1. **Convert manifest** to v2 format (Firefox requirement)
2. **Submit to addons.mozilla.org**
3. **Free submission** process

### **Edge Add-ons**
1. **Same as Chrome** (Manifest v3 compatible)
2. **Submit to Microsoft Edge Add-ons**

---

## Troubleshooting

### **Common Issues**

**Extension not loading**
- Check `chrome://extensions/` for errors
- Verify all files are in `frontend/` folder
- Check `manifest.json` syntax

**"Failed to contact backend"**
- Verify backend URL in `content.js`
- Check if backend server is running
- Test backend with: `curl http://localhost:5001/model_info`

**Not detecting YouTube videos**
- Check browser console (F12) for JavaScript errors
- Verify YouTube URL patterns in `content.js`
- Test on different video types (normal videos vs Shorts)

**UI not showing**
- Check CSS conflicts with YouTube
- Verify overlay z-index is high enough
- Test on different YouTube layouts

### **Debug Mode**
Add console logging to `content.js`:

```javascript
// Add debugging
console.log('Jumpscare detector: Video URL changed to', url);
console.log('Jumpscare detector: Backend response', data);
```

---

## 📱 Browser Compatibility

| Browser | Manifest | Status | Notes |
|---------|----------|--------|-------|
| **Chrome** | v3 | Full support | Primary target |
| **Edge** | v3 | Full support | Same as Chrome |
| **Firefox** | v2 | ⚠️ Needs conversion | Different manifest format |
| **Safari** | Safari format | ❓ Possible | Requires Safari-specific conversion |

---

## Extension Permissions

```json
{
  "permissions": ["activeTab", "scripting"],
  "host_permissions": ["https://www.youtube.com/*"]
}
```

- **`activeTab`**: Access current tab when extension activated
- **`scripting`**: Inject content scripts into YouTube
- **`youtube.com`**: Only works on YouTube (privacy-focused)

**No excessive permissions** - respects user privacy! 🔒

---

**🎬 Enjoy safe YouTube browsing!** 