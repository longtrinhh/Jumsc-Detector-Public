# ML-Based Jumpscare Detection Setup

## Overview
This approach uses machine learning to detect jumpscares instead of hand-tuned rules. It's more robust and can learn patterns from data.

## Setup Steps

### 1. Install Dependencies
```bash
pip install -r requirements_ml.txt
```

### 2. Collect Training Data
Create a JSON file with labeled examples:

```json
{
  "https://youtube.com/watch?v=scary_video_1": {
    "jumpscares": [12.5, 45.2, 67.8],
    "audio_file": "audio/scary_video_1.wav"
  },
  "https://youtube.com/watch?v=normal_video_1": {
    "jumpscares": [],
    "audio_file": "audio/normal_video_1.wav"
  }
}
```

**Tips for collecting data:**
- Download 20-50 videos (mix of horror and normal content)
- Manually identify jumpscare timestamps
- Use `yt-dlp -f bestaudio -o "audio/%(title)s.wav" URL` to download audio

### 3. Train the Model
```bash
python train_model.py
```

This will:
- Extract audio features (MFCCs, spectral features, etc.)
- Train multiple models (Random Forest, Gradient Boosting, Neural Network)
- Save the best model as `.pkl` files
- Show performance metrics

### 4. Run ML-Powered Backend
```bash
python backend_ml.py
```

The server runs on port 5001 and includes:
- ML-based detection with confidence scores
- Fallback to rule-based if model fails
- Visualization of predictions
- Model retraining API

### 5. Update Extension (Optional)
Change the backend URL in `content.js`:
```javascript
const resp = await fetch('http://localhost:5001/analyze', {
```

## Features

### Adaptive Thresholds
Send custom detection sensitivity:
```javascript
fetch('http://localhost:5001/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    url: video_url,
    threshold: 0.8  // Higher = more strict
  })
})
```

### Model Information
Check model status:
```bash
curl http://localhost:5001/model_info
```

### Retraining
Reload updated model:
```bash
curl -X POST http://localhost:5001/retrain
```

## Model Performance

The system extracts **79 audio features** per 3-second segment:
- **4 RMS features** (mean, std, max, min)
- **26 MFCC features** (13 coefficients × 2 stats each)
- **24 Chroma features** (12 notes × 2 stats each)
- **14 Spectral contrast features** (7 bands × 2 stats each)
- **6 Spectral features** (centroid, rolloff, zero-crossing rate)
- **3 Temporal features** (rate of change statistics)

Expected performance with good training data:
- **Accuracy**: 85-95%
- **False Positive Rate**: <5%
- **Detection Latency**: ~1-2 seconds

## Troubleshooting

### "No trained model available"
Run `python train_model.py` first with proper training data.

### Poor detection accuracy
- Add more diverse training examples
- Check audio quality (should be clear, not compressed)
- Adjust detection threshold (0.5-0.9 range)
- Verify jumpscare timestamps are accurate

### Memory issues
- Reduce audio sample rate in `data_collector.py`
- Use smaller feature sets
- Process shorter audio segments

## Advanced Usage

### Custom Features
Modify `extract_audio_features()` in `data_collector.py` to add:
- Onset detection features
- Harmonic/percussive separation
- Tempo/beat tracking
- Loudness range (LUFS)

### Deep Learning
For larger datasets (>1000 examples), consider:
- CNN on spectrograms
- LSTM for temporal patterns
- Pre-trained audio models (VGGish, OpenL3)

### Model Ensemble
Train multiple models and average predictions for better accuracy. 