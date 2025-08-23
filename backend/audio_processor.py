import os
import time
import uuid
import asyncio
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Memory-optimized audio processing for voice cloning"""
    
    def __init__(self, base_path: str = "/Users/nirajdesai/Documents/AI/voice-clone"):
        self.base_path = Path(base_path)
        self.processed_dir = self.base_path / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio processing settings optimized for M1 MacBook 8GB RAM
        self.target_sr = 22050  # 22.05kHz for TTS compatibility
        self.min_duration = 15  # seconds (reduced from 30 for memory efficiency)
        self.max_duration = 30  # seconds (reduced from 45 for memory efficiency)
        
    def validate_duration(self, duration: float) -> Tuple[bool, Optional[str]]:
        """Validate audio clip duration"""
        if duration < self.min_duration:
            return False, f"Clip too short ({duration:.1f}s). Please record between 15-30 seconds per clip."
        elif duration > self.max_duration:
            return False, f"Clip too long ({duration:.1f}s). Please record between 15-30 seconds per clip."
        return True, None
    
    async def process_audio_file(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Process uploaded audio file: convert to 22.05kHz mono, normalize, denoise
        Optimized for memory efficiency on M1 MacBook
        """
        try:
            # Generate unique clip ID
            clip_id = f"clip_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            
            # Write temporary file
            temp_path = self.processed_dir / f"temp_{clip_id}.wav"
            with open(temp_path, "wb") as f:
                f.write(file_data)
            
            # Load audio with librosa (memory efficient)
            try:
                # Load at target sample rate directly to save memory
                audio, sr = librosa.load(str(temp_path), sr=self.target_sr, mono=True)
            except Exception as e:
                os.remove(temp_path)
                return {"error": f"Failed to load audio file: {str(e)}"}
            
            # Validate duration
            duration = len(audio) / sr
            is_valid, error_msg = self.validate_duration(duration)
            if not is_valid:
                os.remove(temp_path)
                return {"error": error_msg}
            
            # Optimize audio processing in chunks to manage memory
            chunk_size = sr * 10  # 10-second chunks
            processed_chunks = []
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                # Denoise chunk (reduces background noise)
                try:
                    chunk = nr.reduce_noise(y=chunk, sr=sr, stationary=False)
                except:
                    # If denoising fails, continue without it
                    pass
                
                # Normalize volume
                if np.max(np.abs(chunk)) > 0:
                    chunk = chunk / np.max(np.abs(chunk)) * 0.95
                
                processed_chunks.append(chunk)
            
            # Concatenate processed chunks
            processed_audio = np.concatenate(processed_chunks)
            
            # Final output path
            output_path = self.processed_dir / f"{clip_id}.wav"
            
            # Save processed audio
            sf.write(str(output_path), processed_audio, sr)
            
            # Create backup copy in raw folder (safety measure)
            backup_dir = self.base_path / "data" / "raw"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"{clip_id}_backup.wav"
            sf.write(str(backup_path), processed_audio, sr)
            logger.info(f"Audio saved with backup: {output_path} -> {backup_path}")
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Calculate quality score based on various factors
            quality_score = self._calculate_quality_score(processed_audio, sr)
            
            # Count total processed clips
            total_clips = len(list(self.processed_dir.glob("clip_*.wav")))
            
            return {
                "clip_id": clip_id,
                "duration_sec": float(duration),
                "quality_score": quality_score,
                "status": "stored",
                "total_clips": total_clips,
                "target_clips": 20,
                "file_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return {"error": f"Audio processing failed: {str(e)}"}
    
    def _calculate_quality_score(self, audio: np.ndarray, sr: int) -> float:
        """Calculate audio quality score (0-100)"""
        try:
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio ** 2)
            if signal_power == 0:
                return 0.0
            
            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            
            # Spectral centroid (voice clarity indicator)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroid)
            
            # Zero crossing rate (voice activity indicator)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            avg_zcr = np.mean(zcr)
            
            # Combine metrics into quality score
            quality = min(100, max(0, (
                (signal_power * 100) * 0.3 +
                (dynamic_range * 100) * 0.3 +
                (min(avg_centroid / 3000, 1) * 100) * 0.2 +
                (min(avg_zcr * 10, 1) * 100) * 0.2
            )))
            
            return round(quality, 1)
            
        except Exception:
            return 50.0  # Default score if calculation fails
    
    def get_processed_clips(self) -> list:
        """Get list of all processed clips"""
        clips = []
        for clip_path in self.processed_dir.glob("clip_*.wav"):
            try:
                audio, sr = librosa.load(str(clip_path), sr=None, duration=1)  # Just get metadata
                duration = librosa.get_duration(path=str(clip_path))
                clips.append({
                    "path": str(clip_path),
                    "clip_id": clip_path.stem,
                    "duration": duration,
                    "sample_rate": sr
                })
            except Exception as e:
                logger.warning(f"Skipping invalid clip {clip_path}: {e}")
                continue
        return clips
    
    def cleanup_temp_files(self):
        """Clean up any temporary files"""
        for temp_file in self.processed_dir.glob("temp_*.wav"):
            try:
                os.remove(temp_file)
            except:
                pass
