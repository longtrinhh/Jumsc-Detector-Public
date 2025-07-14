import os
import sqlite3
import librosa
import numpy as np
from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
from urllib.parse import urlparse, parse_qs, urlunparse
import uuid
from flask import after_this_request

app = Flask(__name__)
CORS(app)
DB_PATH = 'jumpscares.db'

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
        conn.execute('REPLACE INTO jumpscares (url, timestamps) VALUES (?, ?)', (url, ','.join(map(str, timestamps))))

def get_jumpscares(url):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute('SELECT timestamps FROM jumpscares WHERE url=?', (url,))
        row = cur.fetchone()
        if row and row[0]:
            # Only convert non-empty strings to float
            return [float(t) for t in row[0].split(',') if t.strip() != '']
        else:
            return []

# --- Audio analysis ---
def download_audio(url, out_path):
    subprocess.run([
        'yt-dlp', '-f', 'bestaudio', '-x', '--audio-format', 'wav', '-o', out_path, url
    ], check=True)

def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(sr * 0.05)  # 50ms window
    hop_length = int(sr * 0.025)   # 25ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Smoothing
    smooth_rms = np.convolve(rms, np.ones(3)/3, mode='same')

    jumpscares = []
    
    # Much more conservative thresholds
    baseline_rms = np.percentile(smooth_rms, 30)  # 30th percentile as baseline
    quiet_threshold = np.percentile(smooth_rms, 8)  # 8th percentile as quiet (very conservative)
    
    # Much stricter jump detection
    min_jump_magnitude = max(0.04, baseline_rms * 4.0)  # Increased to 4x baseline, min 0.04
    min_quiet_duration = 0.5  # Keep at 0.5 seconds
    window = int(min_quiet_duration / (hop_length / sr))
    
    # Much higher spike threshold
    spike_threshold_multiplier = 8.0  # Increased from 5x to 8x baseline
    
    print(f"baseline_rms={baseline_rms:.4f}, quiet_threshold={quiet_threshold:.4f}, min_jump_magnitude={min_jump_magnitude:.4f}")

    idx = window
    last_detection_time = -10.0  # Initialize to allow first detection
    
    while idx < len(smooth_rms):
        time = times[idx]
        
        # Skip if too close to last detection (minimum 3 seconds between detections)
        if time - last_detection_time < 3.0:
            idx += 1
            continue
            
        pre_rms = smooth_rms[idx-window:idx]
        post_rms = smooth_rms[idx]
        
        # Calculate pre-window statistics
        pre_mean = np.mean(pre_rms)
        pre_min = np.min(pre_rms)
        
        # Very restrictive quiet detection
        quiet_frames = np.sum(pre_rms <= quiet_threshold)
        is_quiet_period = (quiet_frames >= window * 0.9)  # 90% of frames should be quiet (very restrictive)
        
        # Stricter quiet moment check
        has_quiet_moment = pre_min <= quiet_threshold and pre_mean <= baseline_rms * 1.2
        
        # Calculate jump magnitude
        jump_magnitude = post_rms - pre_mean
        
        # Multi-frame jump detection with longer lookahead
        lookahead = min(10, len(smooth_rms) - idx)
        peak_rms = np.max(smooth_rms[idx:idx+lookahead])
        peak_jump = peak_rms - pre_mean
        
        # ONLY allow positive jumps
        if jump_magnitude <= 0 and peak_jump <= 0:
            idx += 1
            continue
            
        # More restrictive detection conditions
        significant_jump = (jump_magnitude > min_jump_magnitude or peak_jump > min_jump_magnitude)
        sudden_spike = (post_rms > baseline_rms * spike_threshold_multiplier)  # Very high spike
        
        # Additional filter: must be a significant increase from baseline
        above_baseline = (post_rms > baseline_rms * 3.0)  # Increased from 2x to 3x
        
        # Even more restrictive combined conditions
        if (
            (is_quiet_period or has_quiet_moment) and
            significant_jump and
            above_baseline and
            time > 3.0 and
            jump_magnitude > 0.02  # Must be at least 0.02 absolute jump
        ) or (sudden_spike and time > 3.0 and jump_magnitude > 0.03):  # Sudden spike needs larger jump
            jumpscares.append(time)
            last_detection_time = time
            print(f"Jumpscare detected at {time:.2f}s: pre_mean={pre_mean:.4f}, post_rms={post_rms:.4f}, jump={jump_magnitude:.4f}, peak_jump={peak_jump:.4f}")
            idx += int(window * 3.0)  # Skip ahead much more to avoid multiple detections
        else:
            idx += 1

    # Enhanced debugging
    max_rms = np.max(smooth_rms)
    max_idx = np.argmax(smooth_rms)
    max_time = times[max_idx]
    
    print(f"Max RMS: {max_rms:.4f} at {max_time:.2f}s")
    print(f"Overall RMS stats: min={np.min(smooth_rms):.4f}, mean={np.mean(smooth_rms):.4f}, max={max_rms:.4f}")
    print("Detected jumpscares at:", jumpscares)

    plt.figure(figsize=(12, 6))
    plt.plot(times, smooth_rms, 'b-', linewidth=1)
    plt.axhline(y=quiet_threshold, color='r', linestyle='--', label=f'Quiet threshold: {quiet_threshold:.4f}')
    plt.axhline(y=baseline_rms, color='g', linestyle='--', label=f'Baseline: {baseline_rms:.4f}')
    plt.axhline(y=baseline_rms * spike_threshold_multiplier, color='orange', linestyle=':', label=f'Spike threshold: {baseline_rms * spike_threshold_multiplier:.4f}')
    plt.axhline(y=baseline_rms * 3.0, color='purple', linestyle='-.', label=f'Min level: {baseline_rms * 3.0:.4f}')
    plt.axhline(y=min_jump_magnitude, color='cyan', linestyle=':', label=f'Min jump: {min_jump_magnitude:.4f}')
    
    # Mark detected jumpscares
    for js_time in jumpscares:
        plt.axvline(x=js_time, color='red', linestyle='-', alpha=0.7, label=f'Jumpscare: {js_time:.1f}s')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed RMS")
    plt.title("Audio RMS over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rms_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved RMS plot to rms_plot.png")

    return [float(t) for t in jumpscares]

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
    return url  # fallback

# --- API endpoints ---
@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    existing = get_jumpscares(url)
    if existing:
        return jsonify({'url': url, 'jumpscares': existing, 'cached': True})

    unique_id = str(uuid.uuid4())
    out_path = f'temp_audio_{unique_id}.wav'
    try:
        clean_url = clean_youtube_url(url)
        download_audio(clean_url, out_path)
        audio_file = out_path
        if not os.path.exists(audio_file):
            raise Exception('Audio file not found after download.')
        jumpscares = analyze_audio(audio_file)
        save_jumpscares(url, jumpscares)
        @after_this_request
        def cleanup(response):
            threading.Thread(target=delete_files_later, args=([audio_file],)).start()
            return response
        _ = cleanup
        return jsonify({'url': url, 'jumpscares': jumpscares, 'cached': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/jumpscares', methods=['GET'])
def get_jumpscares_api():
    url = request.args.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    jumpscares = get_jumpscares(url)
    return jsonify({'url': url, 'jumpscares': jumpscares})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)