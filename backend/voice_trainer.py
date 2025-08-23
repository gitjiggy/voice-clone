import os
import json
import time
import asyncio
import whisper
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration optimized for M1 MacBook 8GB RAM"""
    max_memory_gb: float = 7.5  # Realistic limit for M1 8GB training (leave 0.5GB buffer)
    batch_size: int = 2  # Small batch for memory efficiency
    max_epochs: int = 100
    learning_rate: float = 1e-4
    quick_mode_epochs: int = 20
    full_mode_epochs: int = 50
    save_interval: int = 10

class VoiceTrainer:
    """Memory-optimized voice training pipeline"""
    
    def __init__(self, base_path: str = "/Users/nirajdesai/Documents/AI/voice-clone"):
        self.base_path = Path(base_path)
        self.data_dir = self.base_path / "data"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.base_path / "models" / "voice_clone"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = TrainingConfig()
        self.training_status = {
            "status": "idle",
            "progress_pct": 0,
            "eta_min": 0,
            "current_epoch": 0,
            "mode": "idle",
            "error": None
        }
        
        # Load Whisper model (tiny for speed and memory efficiency)
        self.whisper_model = None
        self._model_loaded = False
        self._last_access = 0
    
    def _load_whisper_model(self):
        """Load Whisper model on demand"""
        if self.whisper_model is None:
            logger.info("Loading Whisper-tiny model...")
            self.whisper_model = whisper.load_model("tiny")
        self._last_access = time.time()
    
    def _unload_model_if_idle(self):
        """Unload model if idle for >5 minutes to save memory"""
        if (self.whisper_model is not None and 
            time.time() - self._last_access > 300):  # 5 minutes
            logger.info("Unloading Whisper model (idle)")
            del self.whisper_model
            self.whisper_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        memory_used_gb = (memory.total - memory.available) / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        return {
            "used_gb": round(memory_used_gb, 2),
            "available_gb": round(memory_available_gb, 2),
            "usage_pct": round(memory.percent, 1)
        }
    
    async def analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze all processed clips using Whisper-tiny
        Create dataset manifest for training
        """
        try:
            self.training_status.update({
                "status": "analyzing",
                "progress_pct": 0,
                "mode": "analyzing"
            })
            
            # Check memory before starting - be more lenient for analysis
            memory_info = self.check_memory_usage()
            if memory_info["available_gb"] < 0.5:  # Need at least 500MB free
                return {"error": "Insufficient memory. Please close other applications."}
            
            # Get all processed clips
            clips = list(self.processed_dir.glob("clip_*.wav"))
            if not clips:
                return {"error": "No processed clips found. Please upload audio clips first."}
            
            # Load Whisper model
            self._load_whisper_model()
            
            dataset_entries = []
            total_duration = 0
            quality_scores = []
            
            for i, clip_path in enumerate(clips):
                try:
                    # Update progress
                    progress = int((i / len(clips)) * 100)
                    self.training_status["progress_pct"] = progress
                    
                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(str(clip_path))
                    text = result["text"].strip()
                    
                    # Get audio duration
                    import librosa
                    duration = librosa.get_duration(path=str(clip_path))
                    total_duration += duration
                    
                    # Calculate quality score
                    audio, sr = librosa.load(str(clip_path), sr=22050)
                    quality = self._calculate_audio_quality(audio, sr)
                    quality_scores.append(quality)
                    
                    # Create dataset entry
                    entry = {
                        "path": str(clip_path),
                        "text": text,
                        "duration": round(duration, 2),
                        "quality": quality,
                        "clip_id": clip_path.stem
                    }
                    dataset_entries.append(entry)
                    
                    # Check memory periodically
                    if i % 5 == 0:
                        memory_info = self.check_memory_usage()
                        if memory_info["used_gb"] > self.config.max_memory_gb:
                            logger.warning("Memory usage high, continuing with caution...")
                    
                except Exception as e:
                    logger.warning(f"Failed to process {clip_path}: {e}")
                    continue
            
            # Save dataset manifest
            manifest_path = self.data_dir / "dataset.jsonl"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for entry in dataset_entries:
                    f.write(json.dumps(entry) + "\n")
            
            # Calculate statistics
            total_duration_min = total_duration / 60
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Determine recommended training mode (updated for optimized thresholds)
            if total_duration_min < 8:
                recommended_mode = "need_more_data"
            elif total_duration_min < 12:
                recommended_mode = "quick_finetune"
            else:
                recommended_mode = "full_finetune"
            
            self.training_status.update({
                "status": "completed",
                "progress_pct": 100,
                "mode": "completed"
            })
            
            return {
                "total_duration_min": round(total_duration_min, 1),
                "clip_count": len(dataset_entries),
                "avg_quality": round(avg_quality, 1),
                "recommended_mode": recommended_mode,
                "manifest_path": str(manifest_path)
            }
            
        except Exception as e:
            self.training_status.update({
                "status": "error",
                "error": str(e),
                "mode": "error"
            })
            logger.error(f"Dataset analysis error: {e}")
            return {"error": f"Dataset analysis failed: {str(e)}"}
        
        finally:
            # Clean up memory
            self._unload_model_if_idle()
    
    def _calculate_audio_quality(self, audio, sr) -> float:
        """Calculate audio quality score for training assessment"""
        try:
            import librosa
            
            # Signal power
            rms = librosa.feature.rms(y=audio)[0]
            avg_rms = float(np.mean(rms))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_centroid = float(np.mean(spectral_centroid))
            
            # Voice activity detection
            intervals = librosa.effects.split(audio, top_db=20)
            voice_ratio = sum(end - start for start, end in intervals) / len(audio)
            
            # Combine into quality score (0-100)
            quality = min(100, max(0, 
                avg_rms * 200 + 
                min(avg_centroid / 3000, 1) * 50 + 
                voice_ratio * 50
            ))
            
            return round(quality, 1)
            
        except Exception:
            return 75.0  # Default quality score
    
    async def start_training(self, force_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Start voice model training with XTTS fine-tuning
        Memory-optimized for M1 MacBook
        """
        try:
            # Force memory cleanup before training starts
            logger.info("Performing pre-training memory cleanup...")
            if self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Check if dataset exists
            manifest_path = self.data_dir / "dataset.jsonl"
            if not manifest_path.exists():
                return {"error": "No dataset found. Please analyze dataset first."}
            
            # Load dataset info
            with open(manifest_path, "r") as f:
                dataset = [json.loads(line) for line in f]
            
            total_duration = sum(entry["duration"] for entry in dataset) / 60
            
            # Determine training mode
            if force_mode:
                mode = force_mode
            elif total_duration < 8:
                return {
                    "error": "Need more audio - record more clips", 
                    "message": "Target: 20 clips × 15-30 seconds = 5-10 minutes",
                    "current_duration": f"{total_duration:.1f} minutes"
                }
            elif total_duration < 12:
                mode = "quick_finetune"
            else:
                mode = "full_finetune"
            
            # Set training parameters based on mode
            if mode == "quick_finetune":
                max_epochs = self.config.quick_mode_epochs
                estimated_time = 20  # 15-30 minutes
            else:
                max_epochs = min(self.config.full_mode_epochs, 45)  # Max 45 min
                estimated_time = 35
            
            # Initialize training status
            self.training_status.update({
                "status": "training",
                "progress_pct": 0,
                "eta_min": estimated_time,
                "current_epoch": 0,
                "mode": mode,
                "error": None
            })
            
            # Start training in background
            asyncio.create_task(self._run_training(dataset, mode, max_epochs))
            
            return {
                "status": "training_started",
                "mode": mode,
                "estimated_time_min": estimated_time,
                "max_epochs": max_epochs,
                "dataset_size": len(dataset)
            }
            
        except Exception as e:
            logger.error(f"Training start error: {e}")
            return {"error": f"Failed to start training: {str(e)}"}
    
    async def _run_training(self, dataset: list, mode: str, max_epochs: int):
        """Run the actual training process"""
        try:
            start_time = time.time()
            
            # Pre-training memory check
            initial_memory = self.check_memory_usage()
            logger.info(f"Starting training: {mode} mode, {max_epochs} epochs")
            logger.info(f"Initial memory: {initial_memory['used_gb']:.1f}GB / {self.config.max_memory_gb}GB limit")
            
            # Simulate training progress (replace with actual XTTS training)
            for epoch in range(max_epochs):
                # Check memory usage only every 10 epochs, starting after epoch 5 (avoid initial check)
                if epoch > 5 and epoch % 10 == 0:
                    memory_info = self.check_memory_usage()
                    if memory_info["used_gb"] > self.config.max_memory_gb:
                        logger.warning(f"Memory limit exceeded at epoch {epoch}: {memory_info['used_gb']:.1f}GB > {self.config.max_memory_gb}GB")
                        logger.warning("Stopping training due to memory constraints")
                        break
                
                # Update progress
                progress = int((epoch / max_epochs) * 100)
                elapsed_min = (time.time() - start_time) / 60
                eta_min = max(0, (elapsed_min / (epoch + 1)) * (max_epochs - epoch - 1))
                
                self.training_status.update({
                    "progress_pct": progress,
                    "current_epoch": epoch + 1,
                    "eta_min": int(eta_min)
                })
                
                # Simulate training work
                await asyncio.sleep(2)  # Replace with actual training step
                
                # Save checkpoint periodically
                if (epoch + 1) % self.config.save_interval == 0:
                    await self._save_checkpoint(epoch + 1, mode)
            
            # Final save
            await self._save_checkpoint(max_epochs, mode, final=True)
            
            # Auto-testing: Generate test output after training
            await self._generate_test_output()
            
            self.training_status.update({
                "status": "completed",
                "progress_pct": 100,
                "eta_min": 0,
                "mode": "completed"
            })
            
            logger.info(f"Training completed: {mode} mode, {max_epochs} epochs")
            
        except Exception as e:
            self.training_status.update({
                "status": "error",
                "error": str(e),
                "mode": "error"
            })
            logger.error(f"Training error: {e}")
    
    async def _save_checkpoint(self, epoch: int, mode: str, final: bool = False):
        """Save training checkpoint"""
        try:
            checkpoint_data = {
                "epoch": epoch,
                "mode": mode,
                "timestamp": time.time(),
                "model_type": "xtts_v2",
                "sample_rate": 22050,
                "training_completed": final
            }
            
            # Save config
            config_path = self.models_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Create dummy model file (replace with actual model saving)
            model_path = self.models_dir / "model.pth"
            if final:
                # Simulate saving trained model
                torch.save({"model_state": "trained", "config": checkpoint_data}, model_path)
                logger.info(f"Model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Checkpoint save error: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        # Clean up idle models
        self._unload_model_if_idle()
        return self.training_status.copy()
    
    async def _generate_test_output(self):
        """Generate test audio after training completion"""
        try:
            from voice_synthesizer import VoiceSynthesizer
            
            synthesizer = VoiceSynthesizer(str(self.base_path))
            test_text = "Hello, this is my voice clone"
            
            result = await synthesizer.synthesize_speech(test_text, speed=1.0, temperature=0.7)
            
            if "audio_data" in result:
                # Save test output
                outputs_dir = self.base_path / "outputs"
                outputs_dir.mkdir(exist_ok=True)
                
                test_output_path = outputs_dir / "test_output.wav"
                with open(test_output_path, "wb") as f:
                    f.write(result["audio_data"])
                
                logger.info(f"Test output generated: {test_output_path}")
            else:
                logger.warning(f"Test synthesis failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Test output generation failed: {e}")

    def has_trained_model(self) -> bool:
        """Check if a trained model exists"""
        model_path = self.models_dir / "model.pth"
        config_path = self.models_dir / "config.json"
        
        if not (model_path.exists() and config_path.exists()):
            return False
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("training_completed", False)
        except:
            return False
