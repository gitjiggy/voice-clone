import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import librosa
from TTS.api import TTS

logger = logging.getLogger(__name__)

class PretrainedVoiceSynthesizer:
    """
    Voice synthesizer using pre-trained TTS with voice profile adaptation
    Simple and reliable voice conversion
    """
    
    def __init__(self, base_path: str = "/Users/nirajdesai/Documents/AI/voice-clone"):
        self.base_path = Path(base_path)
        self.models_dir = self.base_path / "models" / "voice_clone"
        self.outputs_dir = self.base_path / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        self.tts_model = None
        self.voice_profile = None
        self._model_loaded = False
        self._last_access = 0
    
    def _load_voice_profile(self) -> bool:
        """Load the created voice profile"""
        try:
            profile_path = self.models_dir / "voice_profile.json"
            config_path = self.models_dir / "config.json"
            
            if not (profile_path.exists() and config_path.exists()):
                logger.warning("Voice profile not found")
                return False
            
            # Load config first
            with open(config_path) as f:
                config = json.load(f)
            
            if not config.get("training_completed", False):
                logger.warning("Voice profile not completed")
                return False
            
            # Load voice profile
            with open(profile_path) as f:
                self.voice_profile = json.load(f)
            
            # Load TTS model
            logger.info("🤖 Loading pre-trained TTS model...")
            base_model = self.voice_profile.get('base_model', 'tacotron2-DDC')
            self.tts_model = TTS(f"tts_models/en/ljspeech/{base_model}")
            
            self._model_loaded = True
            self._last_access = time.time()
            
            logger.info("✅ Voice profile loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load voice profile: {e}")
            return False
    
    def _unload_model_if_idle(self):
        """Unload model if idle for >5 minutes to save memory"""
        if (self.tts_model is not None and 
            time.time() - self._last_access > 300):  # 5 minutes
            logger.info("Unloading TTS model (idle)")
            del self.tts_model
            self.tts_model = None
            self._model_loaded = False
    
    def is_model_ready(self) -> bool:
        """Check if voice profile is ready for synthesis"""
        if not self._model_loaded:
            return self._load_voice_profile()
        
        self._unload_model_if_idle()
        return self._model_loaded
    
    async def synthesize_speech(self, text: str, speed: float = 1.0, temperature: float = 0.7) -> bytes:
        """
        Generate speech using pre-trained TTS with voice profile adaptation
        """
        try:
            if not self.is_model_ready():
                raise Exception("Voice profile not ready. Please complete voice profile creation first.")
            
            if not text or len(text.strip()) == 0:
                raise Exception("Text cannot be empty")
            
            if len(text) > 500:
                text = text[:500]  # Limit text length
            
            logger.info(f"🎤 Synthesizing with voice profile: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate base audio with TTS model
            wav = self.tts_model.tts(text=text)
            
            # Convert to numpy array if needed
            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)
            elif not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)
            
            # Apply voice profile characteristics
            if self.voice_profile and 'speaker_characteristics' in self.voice_profile:
                wav = self._apply_voice_characteristics(wav, speed, temperature)
            
            # Convert to bytes and return in expected format
            audio_bytes = self._audio_to_bytes(wav, 22050)
            self._last_access = time.time()
            
            return {
                "audio_data": audio_bytes,
                "duration_sec": len(wav) / 22050,
                "sample_rate": 22050
            }
            
        except Exception as e:
            logger.error(f"❌ Voice synthesis failed: {e}")
            raise Exception(f"Voice synthesis failed: {e}")
    
    def _apply_voice_characteristics(self, audio: np.ndarray, speed: float, temperature: float) -> np.ndarray:
        """Apply voice profile characteristics to generated audio"""
        try:
            characteristics = self.voice_profile['speaker_characteristics']
            
            # Apply pitch adjustment
            avg_pitch = characteristics.get('average_pitch', 200)
            pitch_scale = avg_pitch / 200  # Normalize around 200Hz base
            
            # Limit pitch changes to reasonable range
            pitch_scale = np.clip(pitch_scale, 0.7, 1.5)
            
            if abs(pitch_scale - 1.0) > 0.05:  # Only apply if significant difference
                n_steps = np.log2(pitch_scale) * 12  # Convert to semitones
                audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=n_steps)
            
            # Apply energy/volume adjustment
            avg_energy = characteristics.get('average_energy', 0.1)
            energy_scale = np.clip(avg_energy * 8, 0.5, 2.0)  # Scale and limit
            audio = audio * energy_scale
            
            # Apply speed adjustment
            if speed != 1.0:
                speed_factor = np.clip(speed, 0.5, 2.0)
                audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            
            # Apply temperature (expressiveness) via slight pitch variation
            if temperature != 0.7:
                # Higher temperature = more pitch variation
                temp_scale = (temperature - 0.7) * 0.1  # Small variations
                if abs(temp_scale) > 0.02:
                    pitch_variation = np.random.normal(0, abs(temp_scale), len(audio))
                    # Apply subtle pitch modulation (simplified)
                    audio = audio * (1 + pitch_variation * 0.1)
            
            # Normalize to prevent clipping
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
            logger.info(f"✅ Applied voice characteristics: pitch={pitch_scale:.2f}, energy={energy_scale:.2f}")
            return audio
            
        except Exception as e:
            logger.warning(f"Failed to apply voice characteristics: {e}")
            return audio  # Return original audio if processing fails
    
    def _audio_to_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes"""
        try:
            # Ensure audio is in correct format
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Use BytesIO for in-memory conversion
            import io
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()
            buffer.close()
            
            return audio_bytes
                
        except Exception as e:
            logger.error(f"Failed to convert audio to bytes: {e}")
            raise Exception("Audio conversion failed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for health checks"""
        return {
            "model_loaded": self._model_loaded,
            "model_type": "pretrained_tts",
            "voice_profile_ready": self.is_model_ready()
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.tts_model is not None:
            del self.tts_model
            self.tts_model = None
        self._model_loaded = False
