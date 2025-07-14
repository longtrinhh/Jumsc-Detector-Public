# Backend - Jumpscare Detection API

**Backend that analyzes YouTube videos for jumpscare detection.**

## Quick Start

### **1. Install Dependencies**
```bash
pip install -r requirements_ml.txt
```

### **2. Train Models (First Time)**
```bash
python train_model.py
```

### **3. Start Server**
```bash
# Local development
python backend_ml.py

# VPS/Production (see VPS_SETUP.md)
python backend_ml.py
```

---

## Files Overview

### **Core Backend**
- `backend_ml.py` - **Main API server**
- `backend.py` - Rule-based fallback server
- `requirements_ml.txt` - Python dependencies

### **Machine Learning**
- `train_model.py` - **Model training script**
- `data_collector.py` - Audio feature extraction
- `*.pkl` - **Trained models** (Random Forest, Gradient Boosting, Neural Network)
- `video_labels.json` - Training data labels

### **🛠 Utilities**
- `convert_audio.py` - Audio format conversion
- `jumpscares.db` - Results cache database

### **📚 Documentation**
- `VPS_SETUP.md` - **Complete VPS deployment guide**
- `ML_SETUP.md` - ML training documentation

---

## 🔄 API Endpoints

### **POST `/analyze`**
Analyze a YouTube video for jumpscares.

```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "threshold": 0.4
  }'
```

**Response:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "jumpscares": [15.2, 42.8, 67.5],
  "cached": false,
  "method": "machine_learning",
  "threshold_used": 0.4
}
```

### **GET `/model_info`**
Get information about the loaded ML model.

```bash
curl http://localhost:5001/model_info
```

### **POST `/retrain`**
Reload/retrain the ML model.

```bash
curl -X POST http://localhost:5001/retrain
```

---

## Configuration

### **Threshold Settings**
- **0.2-0.3**: Very sensitive (detects almost everything)
- **0.4**: **Default** - Good balance
- **0.5-0.6**: Conservative (only obvious jumpscares)
- **0.7+**: Very strict

### **Server Settings**
```python
# In backend_ml.py
app.run(
    host='0.0.0.0',    # VPS: Allow external connections
    port=5001,         # API port
    debug=False        # Production: disable debug
)
```

---

## Development

### **Local Testing**
```bash
# Start server
python backend_ml.py

# Test in another terminal
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=test"}'
```

### **Training New Models**
1. **Update training data** in `video_labels.json`
2. **Convert audio files**: `python convert_audio.py`
3. **Train models**: `python train_model.py`
4. **Restart server** to load new models

### **Model Performance**
- **Accuracy**: 85-95% with good training data
- **Features**: 79 audio features per 3-second segment
- **Speed**: ~2x faster than 1-second analysis

---

## 🌐 VPS Deployment

### **Quick Deploy**
```bash
# Upload to VPS
scp -r backend/ user@your-vps:/home/user/

# On VPS
cd backend/
pip install -r requirements_ml.txt
python backend_ml.py
```

### **Production Setup**
See `VPS_SETUP.md` for complete production deployment with:
- Systemd service
- Firewall configuration
- SSL/HTTPS setup
- Process monitoring

---

## Troubleshooting

### **Common Issues**

**"No trained model available"**
```bash
python train_model.py  # Train models first
```

**"Audio format not supported"**
```bash
python convert_audio.py  # Convert audio files
```

**"Connection refused"**
- Check if server is running: `ps aux | grep python`
- Verify port 5001 is open: `netstat -tulpn | grep 5001`
- For VPS: Check firewall rules

**"CORS errors"**
- Backend already configured for CORS
- Check browser console for specific errors

### **Performance Optimization**
- **Faster analysis**: Increase `step_size` in `analyze_full_audio()`
- **Memory usage**: Use smaller models or reduce audio quality
- **Caching**: Results automatically cached in `jumpscares.db`

---

## Model Architecture

### **Feature Extraction (79 features)**
- **4 RMS features** (mean, std, max, min)
- **26 MFCC features** (13 coefficients × 2 stats)
- **24 Chroma features** (12 notes × 2 stats)  
- **14 Spectral contrast** (7 bands × 2 stats)
- **6 Spectral features** (centroid, rolloff, zero-crossing)
- **3 Temporal features** (rate of change stats)

### **Models Available**
1. **Random Forest** - Most reliable, good accuracy
2. **Gradient Boosting** - Fast, balanced performance
3. **Neural Network** - Complex patterns, higher accuracy

---

## API Integration

### **Frontend Connection**
```javascript
// In frontend/content.js
const response = await fetch('http://YOUR_VPS_IP:5001/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    url: videoUrl,
    threshold: 0.4 
  })
});
```

### **Custom Applications**
The API can be used by any application that needs jumpscare detection:
- Web applications
- Mobile apps  
- Desktop software
- Other browser extensions

---

**🎉 Your ML backend is ready to detect jumpscares!** 