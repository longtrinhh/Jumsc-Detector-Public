import os
import sqlite3
import librosa
import numpy as np
from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS
import threading
from urllib.parse import urlparse, parse_qs, urlunparse
import uuid
from flask import after_this_request
from train_model import JumpscareDetector
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import asyncio
import queue
import time
import warnings

# Suppress librosa warnings about empty frequency sets
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# Fix matplotlib for web server environment
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

app = Flask(__name__)
# Enhanced CORS for VPS deployment
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
DB_PATH = 'jumpscares.db'

# Global model - load once at startup
detector = None

# Concurrency settings
MAX_WORKERS = 4  # Number of parallel workers
CHUNK_COUNT = 2  # Number of chunks to split video into (2 = half/half)
REQUEST_QUEUE = queue.Queue()  # Queue for pending requests
PROCESSING_LOCK = threading.Lock()  # Lock for shared resources

def load_detector():
    global detector
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different model types in order of preference
    model_files = [
        "jumpscare_model_random_forest.pkl",      # Best general performance
        "jumpscare_model_gradient_boost.pkl",    # Good alternative 
        "jumpscare_model_neural_network.pkl"     # Deep learning option
    ]
    
    for model_filename in model_files:
        model_path = os.path.join(script_dir, model_filename)
        
        if os.path.exists(model_path):
            try:
                detector = JumpscareDetector()
                detector.load_model(model_path)
                print(f"Loaded model: {model_filename}")
                print(f"Model path: {model_path}")
                return  # Success! Exit the function
            except Exception as e:
                print(f"Failed to load {model_filename}: {e}")
                continue
    
    # If we get here, no models could be loaded
    print("No trained models found or loadable!")
    print(f"Searched in directory: {script_dir}")
    print("Available files:")
    for file in os.listdir(script_dir):
        if file.endswith('.pkl'):
            print(f"   {file}")
    print("Please train a model first using train_model.py")
    detector = None

# --- Database setup ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS jumpscares (
                url TEXT PRIMARY KEY,
                timestamps TEXT
            )
        ''')

def save_jumpscares(url, timestamps):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('REPLACE INTO jumpscares (url, timestamps) VALUES (?, ?)', 
                    (url, ','.join(map(str, timestamps))))
        print(f"💾 Database: Saved {len(timestamps)} jumpscares for {url}")

def get_jumpscares(url):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT timestamps FROM jumpscares WHERE url=?', (url,))
        row = cur.fetchone()
        if row and row[0]:
            jumpscares = [float(t) for t in row[0].split(',') if t.strip() != '']
            print(f"Database: Found {len(jumpscares)} jumpscares for {url}")
            return jumpscares
        else:
            print(f"Database: No jumpscares found for {url}")
            return []

# --- Audio analysis with ML ---
def get_cookie_file():
    """Try multiple cookie files for different users"""
    script_dir = os.path.dirname(__file__)
    cookie_files = [
        os.path.join(script_dir, 'youtube_cookies.txt'),           # Your local PC cookies
        os.path.join(script_dir, 'youtube_cookies_pc2.txt'),       # Other PC cookies
        os.path.join(script_dir, 'youtube_cookies_pc3.txt'),       # Another PC cookies
        os.path.join(script_dir, 'youtube_cookies_backup.txt')     # Backup cookies
    ]
    
    for cookie_file in cookie_files:
        if os.path.exists(cookie_file):
            print(f"🍪 Using cookie file: {os.path.basename(cookie_file)}")
            return cookie_file
    
    print("⚠️ No cookie files found, will use browser cookies")
    return None

def download_audio(url, out_path):
    """Download audio with anti-bot protection using exported cookies"""
    
    # Get the best available cookie file
    cookies_file = get_cookie_file()
    
    # Multiple strategies to avoid YouTube bot detection
    strategies = []
    
    # Strategy 1: Use exported cookies file (PRIORITY - most reliable)
    if cookies_file:
        strategies.append([
            'yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'wav',
            '--cookies', cookies_file,
            '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            '--add-header', 'Accept-Language:en-US,en;q=0.9',
            '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            '--add-header', 'Sec-Fetch-Dest:document',
            '--add-header', 'Sec-Fetch-Mode:navigate',
            '--add-header', 'Sec-Fetch-Site:none',
            '-o', out_path, url
        ])
    
    # Strategy 2: Use cookies from browser (if available)
    strategies.append([
        'yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'wav', 
        '--cookies-from-browser', 'chrome',  # Try Chrome cookies first
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '-o', out_path, url
    ])
    
    # Strategy 3: Use Firefox cookies as fallback
    strategies.append([
        'yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'wav',
        '--cookies-from-browser', 'firefox',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '-o', out_path, url
    ])
    
    # Strategy 4: No cookies but with realistic user agent and headers
    strategies.append([
        'yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'wav',
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '--add-header', 'Accept-Language:en-US,en;q=0.9',
        '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        '-o', out_path, url
    ])
    
    # Strategy 5: Last resort - basic download
    strategies.append([
        'yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'wav', 
        '-o', out_path, url
    ])
    
    last_error = None
    for i, cmd in enumerate(strategies, 1):
        try:
            print(f"Trying download strategy {i}/{len(strategies)}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"Success with strategy {i}")
                return
            else:
                last_error = result.stderr
                print(f"Strategy {i} failed: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            print(f"Strategy {i} timed out")
            last_error = "Download timed out"
        except Exception as e:
            print(f"Strategy {i} error: {e}")
            last_error = str(e)
    
    # If all strategies failed
    raise Exception(f"All download strategies failed. Last error: {last_error}")

def analyze_audio_ml_concurrent(audio_path, threshold=0.4):
    """Use trained ML model to detect jumpscares with video chunking and concurrent processing"""
    global detector
    
    if detector is None:
        raise Exception("No trained model available. Please train a model first.")
    
    print(f"Analyzing audio with chunked concurrent model (threshold={threshold})...")
    
    # Get audio duration for chunking
    duration = librosa.get_duration(path=audio_path)
    print(f"Audio duration: {duration:.1f}s")
    
    # Split video into chunks for concurrent processing
    chunk_size = duration / CHUNK_COUNT
    chunks = []
    
    for i in range(CHUNK_COUNT):
        start_time = i * chunk_size
        end_time = (i + 1) * chunk_size if i < CHUNK_COUNT - 1 else duration
        chunks.append((start_time, end_time))
    
    print(f"Splitting into {CHUNK_COUNT} chunks:")
    for i, (start, end) in enumerate(chunks):
        print(f"   Chunk {i+1}: {start:.1f}s - {end:.1f}s")
    
    # Process all chunks concurrently
    with ThreadPoolExecutor(max_workers=CHUNK_COUNT) as executor:
        # Submit all chunk analysis tasks
        future_to_chunk = {
            executor.submit(analyze_chunk, audio_path, start_time, end_time, threshold): f"chunk{i+1}"
            for i, (start_time, end_time) in enumerate(chunks)
        }
        
        # Collect results from both chunks
        all_jumpscares = []
        for future in as_completed(future_to_chunk):
            chunk_name = future_to_chunk[future]
            try:
                chunk_jumpscares = future.result()
                print(f"{chunk_name} completed with {len(chunk_jumpscares)} jumpscares")
                all_jumpscares.extend(chunk_jumpscares)
            except Exception as e:
                print(f"{chunk_name} failed: {e}")
    
    # Sort jumpscares by time and remove duplicates
    all_jumpscares = sorted(list(set(all_jumpscares)))
    
    # Post-processing: Remove jumpscares too close together
    filtered_jumpscares = []
    for js_time in all_jumpscares:
        if not filtered_jumpscares or (js_time - filtered_jumpscares[-1]) >= 3.0:
            filtered_jumpscares.append(js_time)
    
    print(f"Chunked analysis detected {len(filtered_jumpscares)} jumpscares: {filtered_jumpscares}")
    return filtered_jumpscares

def analyze_chunk(audio_path, start_time, end_time, threshold):
    """Analyze a specific chunk of audio"""
    print(f"Analyzing chunk: {start_time:.1f}s - {end_time:.1f}s")
    
    # Create segments within this chunk (every 2 seconds)
    segments = list(range(int(start_time), int(end_time - 3), 2))
    print(f"   Processing {len(segments)} segments in chunk...")
    
    # Process segments in this chunk
    chunk_jumpscares = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit segment analysis tasks for this chunk
        future_to_segment = {
            executor.submit(analyze_segment, audio_path, segment_start, threshold): segment_start 
            for segment_start in segments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_segment):
            segment_start = future_to_segment[future]
            try:
                result = future.result()
                if result is not None:
                    chunk_jumpscares.append(result)
            except Exception as e:
                print(f"   Segment {segment_start}s failed: {e}")
    
    print(f"   Chunk completed with {len(chunk_jumpscares)} jumpscares")
    return chunk_jumpscares

def is_valid_audio_segment(audio_path, start_time, duration=3.0):
    """Check if audio segment is valid for analysis"""
    try:
        y, sr = librosa.load(audio_path, offset=start_time, duration=duration, sr=22050)
        # Check if audio has sufficient content (not silent)
        return np.mean(np.abs(y)) > 1e-6
    except:
        return False

def analyze_segment(audio_path, start_time, threshold):
    """Analyze a single audio segment (thread-safe)"""
    try:
        # Skip silent or invalid segments
        if not is_valid_audio_segment(audio_path, start_time):
            return None
            
        with PROCESSING_LOCK:  # Thread-safe model access
            prob = detector.predict_jumpscare_probability(audio_path, start_time)
        
        if prob > threshold:
            return start_time + 1.5  # Middle of 3-second window
        return None
    except Exception as e:
        # Don't print every error to avoid spam, only log occasionally
        if start_time % 10 == 0:  # Log every 10th error
            print(f"⚠️ Segment analysis failed at {start_time}s: {e}")
        return None

def analyze_audio_ml(audio_path, threshold=0.4):
    """Legacy function - now uses concurrent processing"""
    return analyze_audio_ml_concurrent(audio_path, threshold)

def create_prediction_plot(audio_path, detected_jumpscares, threshold):
    """Create a plot showing the ML model's predictions over time"""
    # Use global matplotlib configuration
    
    duration = librosa.get_duration(path=audio_path)
    times = []
    probabilities = []
    
    # Get predictions for every 2 seconds (faster)
    for start_time in range(0, int(duration - 3), 2):
        prob = detector.predict_jumpscare_probability(audio_path, start_time)
        times.append(start_time + 1.5)  # Middle of 3-second window
        probabilities.append(prob)
    
    # Create plot
    plt.figure(figsize=(15, 6))
    plt.plot(times, probabilities, 'b-', alpha=0.7, label='Jumpscare Probability')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Detection Threshold ({threshold})')
    
    # Mark detected jumpscares
    for js_time in detected_jumpscares:
        plt.axvline(x=js_time, color='red', linestyle='-', alpha=0.8, 
                   label='Detected Jumpscare' if js_time == detected_jumpscares[0] else "")
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Jumpscare Probability')
    plt.title('ML Jumpscare Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.savefig("ml_predictions.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved ML prediction plot to ml_predictions.png")

# Fallback rule-based detection (simplified version)
def analyze_audio_fallback(audio_path):
    """Fallback rule-based detection if ML model is not available"""
    print("Using fallback rule-based detection...")
    
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(sr * 0.05)
    hop_length = int(sr * 0.025)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Simple threshold-based detection
    threshold = np.percentile(rms, 95)
    jumpscares = []
    
    for i, rms_val in enumerate(rms):
        if rms_val > threshold and times[i] > 3.0:
            # Avoid duplicates within 3 seconds
            if not jumpscares or (times[i] - jumpscares[-1]) > 3.0:
                jumpscares.append(float(times[i]))
    
    return jumpscares

def delete_files_later(paths, delay=1):
    import time
    time.sleep(delay)
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted {path}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

def clean_youtube_url(url):
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if 'v' in qs:
        clean_query = f"v={qs['v'][0]}"
        clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', clean_query, ''))
        return clean_url
    return url

# --- API endpoints ---
@app.route('/analyze', methods=['POST'])
def analyze():
    """Concurrent video analysis endpoint"""
    global detector
    
    url = request.json.get('url')
    threshold = request.json.get('threshold', 0.4)
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Normalize URL for consistent caching
    clean_url = clean_youtube_url(url)
    print(f"Cache check for URL: {clean_url}")
    
    with sqlite3.connect(DB_PATH) as conn:
        # Check if the video was analyzed before (even if 0 jumpscares)
        cur = conn.execute('SELECT 1 FROM jumpscares WHERE url=?', (clean_url,))
        if cur.fetchone() is not None:
            existing = get_jumpscares(clean_url)
            print(f"Cache HIT! Found {len(existing)} jumpscares for {clean_url}")
            return jsonify({
                'url': clean_url,
                'jumpscares': existing,
                'cached': True,
                'method': 'cached'
            })
        else:
            print(f"Cache MISS for {clean_url}")
    
    # Generate unique ID for this request
    unique_id = str(uuid.uuid4())
    out_path = f'temp_audio_{unique_id}.wav'
    
    try:
        print(f"Starting concurrent analysis for: {url}")
        start_time = time.time()
        
        # Download audio (can be parallel with other requests)
        download_audio(clean_url, out_path)
        
        if not os.path.exists(out_path):
            raise Exception('Audio file not found after download.')
        
        download_time = time.time() - start_time
        print(f"Download completed in {download_time:.2f}s")
        
        # Analyze with concurrent ML processing
        analysis_start = time.time()
        try:
            jumpscares = analyze_audio_ml_concurrent(out_path, threshold)
            method = 'concurrent_machine_learning'
        except Exception as e:
            print(f"ML detection failed: {e}")
            jumpscares = analyze_audio_fallback(out_path)
            method = 'rule_based_fallback'
        
        analysis_time = time.time() - analysis_start
        total_time = time.time() - start_time
        
        # Save results (thread-safe)
        with PROCESSING_LOCK:
            save_jumpscares(clean_url, jumpscares)
        
        # Cleanup in background thread
        @after_this_request
        def cleanup(response):
            threading.Thread(target=delete_files_later, args=([out_path],)).start()
            return response
        
        _ = cleanup
        
        print(f"Analysis completed in {total_time:.2f}s (download: {download_time:.2f}s, analysis: {analysis_time:.2f}s)")
        
        return jsonify({
            'url': clean_url, 
            'jumpscares': jumpscares, 
            'cached': False,
            'method': method,
            'threshold_used': threshold,
            'processing_time': total_time,
            'concurrent_workers': MAX_WORKERS
        })
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """API endpoint to retrain the model with new data"""
    try:
        # This would trigger retraining with updated data
        # For now, just reload the existing model
        load_detector()
        return jsonify({'message': 'Model reloaded successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to reload model: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    global detector
    
    if detector is None:
        return jsonify({
            'model_loaded': False,
            'model_type': None,
            'message': 'No model loaded. Please train a model first.'
        })
    
    return jsonify({
        'model_loaded': True,
        'model_type': detector.model_type,
        'message': 'ML model is ready for predictions'
    })

@app.route('/jumpscares', methods=['GET'])
def get_jumpscares_api():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    # Thread-safe database access
    with PROCESSING_LOCK:
        jumpscares = get_jumpscares(url)
    return jsonify({'url': url, 'jumpscares': jumpscares})

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status and concurrency info"""
    return jsonify({
        'status': 'running',
        'concurrent_workers': MAX_WORKERS,
        'chunk_count': CHUNK_COUNT,
        'model_loaded': detector is not None,
        'model_type': detector.model_type if detector else None,
        'queue_size': REQUEST_QUEUE.qsize() if hasattr(REQUEST_QUEUE, 'qsize') else 0,
        'features': {
            'chunked_analysis': True,
            'concurrent_processing': True,
            'adaptive_thresholds': True,
            'multiple_cookies': True
        }
    })

if __name__ == '__main__':
    init_db()
    load_detector()
    
    if detector is not None:
        print("Model loaded successfully")
    else:
        print("No model available, will use rule-based fallback")
    
    print("Starting concurrent server...")
    print("Multiple requests can now be processed simultaneously!")
    # For VPS deployment: bind to all interfaces
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)  # threaded=True enables concurrent requests 