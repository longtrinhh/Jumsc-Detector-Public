#!/usr/bin/env python3
"""
Convert audio files to a format that librosa can read properly
"""
import os
import subprocess
import sys
from pathlib import Path

def convert_audio_files():
    """Convert all audio files in the audio folder to proper WAV format"""
    
    audio_dir = Path("audio")
    converted_dir = Path("audio_converted")
    
    # Create converted directory
    converted_dir.mkdir(exist_ok=True)
    
    # Get all audio files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")) + \
                  list(audio_dir.glob("*.m4a")) + list(audio_dir.glob("*.mp4"))
    
    if not audio_files:
        print("No audio files found!")
        return False
    
    print(f"Converting {len(audio_files)} audio files...")
    
    converted_count = 0
    failed_count = 0
    
    for i, input_file in enumerate(audio_files, 1):
        output_file = converted_dir / f"{input_file.stem}.wav"
        
        print(f"\nFile {i}/{len(audio_files)}: {input_file.name}")
        
        try:
            # Convert using ffmpeg
            cmd = [
                "ffmpeg", "-i", str(input_file),
                "-ar", "22050",  # Sample rate for librosa
                "-ac", "1",      # Mono audio
                "-y",            # Overwrite output
                str(output_file)
            ]
            
            print(f"   Converting to {output_file.name}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   Success!")
                converted_count += 1
            else:
                print(f"   Failed: {result.stderr.strip()}")
                failed_count += 1
                
        except FileNotFoundError:
            print(f"   FFmpeg not found! Please install FFmpeg first.")
            print(f"      Run: conda install ffmpeg")
            return False
        except Exception as e:
            print(f"   Error: {e}")
            failed_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count}")
    print(f"Failed: {failed_count}")
    
    if converted_count > 0:
        print(f"\nConverted files are in: {converted_dir}")
        print(f"Next step: Update video_labels.json to point to audio_converted/ folder")
        return True
    else:
        return False

def update_labels_file():
    """Update video_labels.json to use converted audio files"""
    import json
    
    try:
        with open("video_labels.json", "r") as f:
            labels = json.load(f)
        
        # Update audio file paths
        updated_labels = {}
        for url, data in labels.items():
            new_data = data.copy()
            old_path = data["audio_file"]
            filename = Path(old_path).name
            new_data["audio_file"] = f"audio_converted/{filename}"
            updated_labels[url] = new_data
        
        # Save updated labels
        with open("video_labels_converted.json", "w") as f:
            json.dump(updated_labels, f, indent=2)
        
        print(f"Created video_labels_converted.json with updated paths")
        return True
        
    except Exception as e:
        print(f"Failed to update labels file: {e}")
        return False

if __name__ == "__main__":
    print("Audio File Converter")
    
    if convert_audio_files():
        if update_labels_file():
            print(f"\nReady for training!")
            print(f"   Use: python train_model.py")
            print(f"   (Make sure to use video_labels_converted.json)")
    else:
        print(f"\nConversion failed. Try alternative methods below.")
        
    print(f"\nAlternative solutions:")
    print(f"   1. Install FFmpeg: conda install ffmpeg")
    print(f"   2. Re-download with: yt-dlp -f bestaudio -x --audio-format wav")
    print(f"   3. Use online converters to convert files to WAV") 