#!/usr/bin/env python3
"""
Test script for audio upload functionality
Creates test audio files and uploads them to the backend
"""

import numpy as np
import soundfile as sf
import requests
import time
from pathlib import Path

def create_test_audio(filename: str, duration: float = 35.0, sample_rate: int = 44100):
    """Create a test audio file with realistic voice-like characteristics"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate complex waveform that mimics voice characteristics
    base_freq = 180  # Typical male voice fundamental frequency
    
    # Create multiple harmonics for voice-like sound
    audio = np.zeros_like(t)
    for harmonic in [1, 2, 3, 4, 5]:
        freq = base_freq * harmonic
        amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
        
        # Add slight frequency modulation for natural variation
        freq_mod = np.sin(t * 2 * np.pi * 0.3) * 5  # 5Hz variation
        audio += amplitude * np.sin(2 * np.pi * (freq + freq_mod) * t)
    
    # Apply envelope to simulate speech patterns
    # Create segments with pauses to simulate words
    segments = int(duration / 3)  # Roughly 3-second segments
    for i in range(segments):
        start = int(i * sample_rate * 3)
        end = min(start + int(sample_rate * 2), len(audio))  # 2 seconds of "speech"
        
        if start < len(audio):
            # Apply envelope to this segment
            segment_len = end - start
            envelope = np.ones(segment_len)
            
            # Fade in/out
            fade_len = int(sample_rate * 0.1)  # 0.1 second fade
            if segment_len > 2 * fade_len:
                envelope[:fade_len] = np.linspace(0, 1, fade_len)
                envelope[-fade_len:] = np.linspace(1, 0, fade_len)
            
            audio[start:end] *= envelope
            
            # Add silence after each segment (except the last)
            if i < segments - 1:
                silence_start = end
                silence_end = min(silence_start + int(sample_rate * 1), len(audio))
                audio[silence_start:silence_end] = 0
    
    # Normalize to prevent clipping
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Save the test file
    sf.write(filename, audio, sample_rate)
    print(f"Created test audio: {filename} ({duration:.1f}s, {sample_rate}Hz)")

def test_upload(filename: str):
    """Test uploading an audio file to the backend"""
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'audio/wav')}
            response = requests.post('http://127.0.0.1:8000/upload', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful: {filename}")
            print(f"   Clip ID: {result.get('clip_id')}")
            print(f"   Duration: {result.get('duration_sec'):.1f}s")
            print(f"   Quality: {result.get('quality_score')}/100")
            print(f"   Total clips: {result.get('total_clips')}")
            return True
        else:
            print(f"❌ Upload failed: {filename}")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {filename} - {e}")
        return False

def main():
    """Run upload tests"""
    print("🎙️ Creating test audio files...")
    
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Create test files with different characteristics
    test_files = [
        ("test_clip_1.wav", 32.0),  # Good duration
        ("test_clip_2.wav", 38.0),  # Good duration
        ("test_clip_3.wav", 42.0),  # Good duration
        ("test_clip_4.wav", 35.0),  # Good duration
        ("test_clip_5.wav", 33.0),  # Good duration
    ]
    
    # Also test edge cases
    edge_cases = [
        ("test_too_short.wav", 25.0),  # Too short
        ("test_too_long.wav", 50.0),   # Too long
    ]
    
    print("\n📤 Testing valid uploads...")
    success_count = 0
    for filename, duration in test_files:
        filepath = test_dir / filename
        create_test_audio(str(filepath), duration)
        if test_upload(str(filepath)):
            success_count += 1
        time.sleep(1)  # Small delay between uploads
    
    print(f"\n✅ Successfully uploaded {success_count}/{len(test_files)} valid clips")
    
    print("\n⚠️ Testing edge cases (should fail)...")
    for filename, duration in edge_cases:
        filepath = test_dir / filename
        create_test_audio(str(filepath), duration)
        test_upload(str(filepath))
        time.sleep(1)
    
    print("\n🩺 Checking final health status...")
    try:
        response = requests.get('http://127.0.0.1:8000/healthz')
        if response.status_code == 200:
            health = response.json()
            print(f"   Total clips: {health.get('total_clips')}")
            print(f"   Total duration: {health.get('total_duration_min'):.1f} minutes")
            print(f"   Training estimate: {health.get('training_time_estimate_min')} minutes")
            print(f"   Recommended clips: {health.get('recommended_clips')}")
        else:
            print(f"Health check failed: {response.status_code}")
    except Exception as e:
        print(f"Health check error: {e}")

if __name__ == "__main__":
    main()
