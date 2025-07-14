import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import sys
import warnings

# Suppress librosa warnings about empty frequency sets
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

def extract_audio_features(audio_path, segment_start, segment_duration=3.0):
    """Extract features from a 3-second audio segment"""
    try:
        y, sr = librosa.load(audio_path, offset=segment_start, duration=segment_duration, sr=22050)
    except Exception as e:
        print(f"⚠️  Warning: Could not load audio segment at {segment_start}s from {audio_path}")
        print(f"   Error: {e}")
        # Return zeros if audio can't be loaded
        return np.zeros(79)  # Return 79 zero features
    
    # Check if audio is mostly silent
    if np.mean(np.abs(y)) < 1e-6:
        # Return zeros for silent segments
        return np.zeros(79)
    
    # Basic features with error handling
    try:
        rms = librosa.feature.rms(y=y)[0]
    except:
        rms = np.zeros(1)
    
    try:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    except:
        spectral_centroid = np.zeros(1)
    
    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    except:
        spectral_rolloff = np.zeros(1)
    
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
    except:
        zcr = np.zeros(1)
    
    # MFCC features (13 coefficients) with error handling
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    except:
        mfccs = np.zeros((13, 1))
    
    # Chroma features with error handling
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    except:
        chroma = np.zeros((12, 1))
    
    # Spectral contrast with error handling
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    except:
        contrast = np.zeros((7, 1))
    
    # Aggregate features (mean, std, max, min)
    features = []
    
    # RMS statistics
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])
    
    # Spectral features
    features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    features.extend([np.mean(zcr), np.std(zcr)])
    
    # MFCC statistics
    for i in range(13):
        features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
    
    # Chroma statistics
    for i in range(12):
        features.extend([np.mean(chroma[i]), np.std(chroma[i])])
    
    # Contrast statistics
    for i in range(7):
        features.extend([np.mean(contrast[i]), np.std(contrast[i])])
    
    # Temporal features (rate of change)
    if len(rms) > 1:
        rms_diff = np.diff(rms)
        features.extend([np.mean(np.abs(rms_diff)), np.std(rms_diff), np.max(np.abs(rms_diff))])
    else:
        features.extend([0, 0, 0])
    
    return np.array(features)

def get_audio_duration_robust(audio_file):
    """Get audio duration with multiple fallback methods"""
    try:
        # Try librosa first
        return librosa.get_duration(path=audio_file)
    except Exception as e1:
        print(f"⚠️  Primary method failed: {e1}")
        try:
            # Try loading just the first second to estimate
            y, sr = librosa.load(audio_file, duration=1.0, sr=22050)
            # If successful, try to get full duration by loading without duration limit
            y_full, sr = librosa.load(audio_file, sr=22050)
            return len(y_full) / sr
        except Exception as e2:
            print(f"Fallback method failed: {e2}")
            print(f"Cannot process {audio_file} - skipping this file")
            return None

def create_training_data(video_labels_file):
    """
    Create training dataset from labeled video data
    
    video_labels_file format (JSON):
    {
        "video_url_1": {
            "jumpscares": [12.5, 45.2, 67.8],  # timestamps in seconds
            "audio_file": "path/to/audio.wav"
        },
        "video_url_2": {
            "jumpscares": [],
            "audio_file": "path/to/audio.wav"
        }
    }
    """
    with open(video_labels_file, 'r') as f:
        labels = json.load(f)
    
    features_list = []
    labels_list = []
    
    total_videos = len(labels)
    video_count = 0
    total_segments = 0
    skipped_videos = 0
    
    print(f"\nFeature Extraction")
    print(f"Processing {total_videos} videos...")
    
    for video_url, data in labels.items():
        video_count += 1
        audio_file = data['audio_file']
        jumpscare_times = data['jumpscares']
        
        print(f"\nVideo {video_count}/{total_videos}: {audio_file}")
        print(f"Jumpscares at: {jumpscare_times if jumpscare_times else 'None'}")
        
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            skipped_videos += 1
            continue
        
        # Get audio duration with robust error handling
        duration = get_audio_duration_robust(audio_file)
        if duration is None:
            skipped_videos += 1
            continue
            
        num_segments = max(1, int(duration - 3))
        print(f"Duration: {duration:.1f}s -> {num_segments} segments")
        
        # Create segments every 1 second
        segment_count = 0
        successful_segments = 0
        for start_time in range(0, int(duration - 3), 1):
            segment_count += 1
            total_segments += 1
            
            # Show progress for longer videos
            if segment_count % 10 == 0 or segment_count == num_segments:
                print(f"   Processing segment {segment_count}/{num_segments} ({start_time}s-{start_time+3}s)", end="")
                sys.stdout.flush()
            
            # Extract features for 3-second segment
            features = extract_audio_features(audio_file, start_time, 3.0)
            
            # Only add if we got valid features (not all zeros)
            if np.any(features):
                # Check if this segment contains a jumpscare
                segment_end = start_time + 3.0
                is_jumpscare = any(start_time <= js_time <= segment_end for js_time in jumpscare_times)
                
                features_list.append(features)
                labels_list.append(1 if is_jumpscare else 0)
                successful_segments += 1
            
            if segment_count % 10 == 0 or segment_count == num_segments:
                print(f" Done")
        
        print(f"Completed: {successful_segments}/{segment_count} segments extracted successfully")
    
    processed_videos = total_videos - skipped_videos
    print(f"\nFeature extraction complete!")
    print(f"Videos processed: {processed_videos}/{total_videos}")
    if skipped_videos > 0:
        print(f"Videos skipped due to format issues: {skipped_videos}")
    print(f"Total segments processed: {len(features_list)}")
    print(f"Jumpscare segments: {sum(labels_list)}")
    print(f"Normal segments: {len(labels_list) - sum(labels_list)}")
    
    if len(features_list) == 0:
        raise Exception("No audio segments could be processed! Check audio file formats.")
    
    return np.array(features_list), np.array(labels_list)

if __name__ == "__main__":
    # Example usage
    print("Audio feature extraction for jumpscare detection")
    print("Feature vector size:", len(extract_audio_features("dummy", 0))) 