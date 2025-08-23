import sys
import time
import json
import asyncio
import platform
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import aiofiles
import logging

# Import our custom modules
from audio_processor import AudioProcessor
from voice_trainer_pretrained import PretrainedVoiceTrainer
from voice_synthesizer_pretrained import PretrainedVoiceSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SpeechRequest(BaseModel):
    text: str
    speed: float = 1.0
    temperature: float = 0.7

app = FastAPI(
    title="Voice Clone API", 
    version="2.0.0",
    description="Memory-optimized voice cloning platform for M1 MacBook"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:4001", "http://localhost:4001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with pre-trained fallback
audio_processor = AudioProcessor()
voice_trainer = PretrainedVoiceTrainer()
voice_synthesizer = PretrainedVoiceSynthesizer()

# Base path for the project
BASE_PATH = Path("/Users/nirajdesai/Documents/AI/voice-clone")

@app.get("/healthz")
async def enhanced_health_check():
    """Enhanced health check with training estimates and memory usage"""
    try:
        # Get memory info
        memory = psutil.virtual_memory()
        memory_available_gb = round(memory.available / (1024**3), 2)
        memory_usage_gb = round((memory.total - memory.available) / (1024**3), 2)
        
        # Get disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = round(disk.free / (1024**3), 2)
        
        # Check processed clips for training estimates
        processed_clips = audio_processor.get_processed_clips()
        total_clips = len(processed_clips)
        total_duration_min = sum(clip["duration"] for clip in processed_clips) / 60
        
        # Estimate training time based on dataset
        if total_duration_min < 15:
            training_time_estimate_min = 0
            recommended_clips = 30 - total_clips
        elif total_duration_min < 20:
            training_time_estimate_min = 25  # Quick fine-tune
            recommended_clips = max(0, 25 - total_clips)
        else:
            training_time_estimate_min = 40  # Full fine-tune
            recommended_clips = 0
        
        # Check if model is loaded
        model_loaded = voice_synthesizer.get_model_info()["model_loaded"]
        
        return {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_version": torch.__version__,
            "memory_available_gb": memory_available_gb,
            "memory_usage_gb": memory_usage_gb,
            "disk_free_gb": disk_free_gb,
            "model_loaded": model_loaded,
            "status": "healthy",
            "platform": platform.platform(),
            # Enhanced fields
            "training_time_estimate_min": training_time_estimate_min,
            "recommended_clips": recommended_clips,
            "total_clips": total_clips,
            "total_duration_min": round(total_duration_min, 1)
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "memory_available_gb": 0,
            "disk_free_gb": 0,
            "model_loaded": False
        }

@app.post("/upload")
async def upload_audio_clip(file: UploadFile = File(...)):
    """
    Accept .webm/.wav via multipart, convert to 22.05kHz mono WAV, normalize volume
    Store in /data/processed/clip_{timestamp}.wav
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.wav', '.webm', '.mp3', '.m4a']:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload .wav, .webm, .mp3, or .m4a files"
            )
        
        # Check file size (limit to 50MB to prevent memory issues)
        file_data = await file.read()
        if len(file_data) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
        
        # Process the audio file
        result = await audio_processor.process_audio_file(file_data, file.filename)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/recording-script")
async def get_recording_script():
    """Return optimized phrases from /scripts/recording_prompts.txt"""
    try:
        prompts_path = BASE_PATH / "scripts" / "recording_prompts.txt"
        
        if not prompts_path.exists():
            raise HTTPException(status_code=404, detail="Recording prompts file not found")
        
        async with aiofiles.open(prompts_path, "r", encoding="utf-8") as f:
            content = await f.read()
        
        # Parse prompts (extract numbered lines)
        lines = content.split('\n')
        prompts = []
        
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and '. ' in line:
                # Extract the prompt text after the number
                prompt_text = line.split('. ', 1)[1]
                prompts.append(prompt_text)
        
        return {
            "prompts": prompts,
            "total_prompts": len(prompts),
            "recommended_clips": 20,
            "instructions": [
                "Record each phrase clearly with consistent tone and pace",
                "Maintain 6-8 inches distance from microphone", 
                "Use quiet environment with minimal background noise",
                "Aim for 15-30 seconds per clip",
                "Pause 2-3 seconds between phrases"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recording script error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load recording script: {str(e)}")

@app.post("/analyze-dataset")
async def analyze_dataset():
    """
    Run Whisper-tiny on all clips, calculate voice consistency score
    Write manifest to data/dataset.jsonl
    """
    try:
        result = await voice_trainer.analyze_dataset()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset analysis failed: {str(e)}")

@app.post("/train")
async def start_training(force_mode: Optional[str] = None):
    """
    Start voice model training with duration-based mode selection
    CRITICAL: Never fallback to synthetic audio - always use real voice cloning
    """
    try:
        result = await voice_trainer.start_training(force_mode=force_mode)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training start error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")

@app.get("/train/status")
async def get_training_status():
    """Get current training status with progress and ETA"""
    try:
        status = voice_trainer.get_training_status()
        return status
        
    except Exception as e:
        logger.error(f"Training status error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "progress_pct": 0,
            "eta_min": 0,
            "current_epoch": 0,
            "mode": "error"
        }

@app.get("/train/progress")
async def stream_training_progress():
    """Stream training progress via Server-Sent Events"""
    async def event_stream():
        try:
            while True:
                status = voice_trainer.get_training_status()
                
                # Send status as SSE
                yield f"data: {json.dumps(status)}\n\n"
                
                # Stop streaming if training is completed or failed
                if status["status"] in ["completed", "error", "idle"]:
                    break
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
        except Exception as e:
            logger.error(f"Progress stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/speak")
async def synthesize_speech(request: SpeechRequest):
    """
    MANDATORY: Only works if a trained voice checkpoint exists
    NO FALLBACKS: If no trained model, return error
    """
    try:
        result = await voice_synthesizer.synthesize_speech(
            text=request.text,
            speed=request.speed,
            temperature=request.temperature
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Return audio as WAV file
        return Response(
            content=result["audio_data"],
            media_type="audio/wav",
            headers={
                "X-Voice-Mode": "trained",
                "X-Duration-Sec": str(result["duration_sec"]),
                "X-Sample-Rate": str(result["sample_rate"]),
                "Content-Disposition": "inline; filename=speech.wav"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the currently loaded model"""
    try:
        model_info = voice_synthesizer.get_model_info()
        training_status = voice_trainer.get_training_status()
        
        return {
            **model_info,
            "training_status": training_status["status"],
            "training_mode": training_status["mode"]
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return {"error": str(e)}

@app.get("/clips")
async def list_processed_clips():
    """List all processed audio clips"""
    try:
        clips = audio_processor.get_processed_clips()
        return {
            "clips": clips,
            "total_clips": len(clips),
            "total_duration_sec": sum(clip["duration"] for clip in clips)
        }
        
    except Exception as e:
        logger.error(f"Clips list error: {e}")
        return {"error": str(e)}

@app.post("/cleanup")
async def cleanup_resources():
    """Clean up memory and temporary files"""
    try:
        audio_processor.cleanup_temp_files()
        voice_synthesizer.cleanup()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {"message": "Cleanup completed"}
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return {"error": str(e)}

@app.post("/reset-training-data")
async def reset_training_data():
    """
    CLEAN SLATE: Remove ALL audio clips, training data, and models for fresh start
    This ensures no leftover data pollutes new training sessions
    """
    try:
        reset_count = {
            "audio_clips": 0,
            "training_files": 0,
            "model_files": 0,
            "output_files": 0,
            "temp_files": 0
        }
        
        base_path = Path("/Users/nirajdesai/Documents/AI/voice-clone")
        
        # 1. Remove all processed audio clips
        processed_dir = base_path / "data" / "processed"
        if processed_dir.exists():
            for clip_file in processed_dir.glob("clip_*.wav"):
                try:
                    clip_file.unlink()
                    reset_count["audio_clips"] += 1
                    logger.info(f"Removed audio clip: {clip_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {clip_file}: {e}")
        
        # 2. Remove training dataset manifest
        dataset_file = base_path / "data" / "dataset.jsonl"
        if dataset_file.exists():
            try:
                dataset_file.unlink()
                reset_count["training_files"] += 1
                logger.info("Removed training dataset manifest")
            except Exception as e:
                logger.warning(f"Failed to remove dataset.jsonl: {e}")
        
        # 3. Remove trained models and configs
        models_dir = base_path / "models" / "voice_clone"
        if models_dir.exists():
            for model_file in models_dir.glob("*"):
                if model_file.is_file():
                    try:
                        model_file.unlink()
                        reset_count["model_files"] += 1
                        logger.info(f"Removed model file: {model_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {model_file}: {e}")
        
        # 4. Remove output audio files
        outputs_dir = base_path / "outputs"
        if outputs_dir.exists():
            for output_file in outputs_dir.glob("*.wav"):
                try:
                    output_file.unlink()
                    reset_count["output_files"] += 1
                    logger.info(f"Removed output file: {output_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {output_file}: {e}")
        
        # 5. Remove temporary files
        if processed_dir.exists():
            for temp_file in processed_dir.glob("temp_*.wav"):
                try:
                    temp_file.unlink()
                    reset_count["temp_files"] += 1
                    logger.info(f"Removed temp file: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {temp_file}: {e}")
        
        # 6. Clear voice trainer status
        voice_trainer.training_status = {
            "status": "idle",
            "progress_pct": 0,
            "eta_min": 0,
            "current_epoch": 0,
            "mode": "idle",
            "error": None
        }
        
        # 7. Cleanup models from memory
        voice_synthesizer.cleanup()
        if hasattr(voice_trainer, 'whisper_model') and voice_trainer.whisper_model:
            del voice_trainer.whisper_model
            voice_trainer.whisper_model = None
        
        # 8. Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        total_removed = sum(reset_count.values())
        
        logger.info(f"Clean slate completed: {total_removed} files removed")
        
        return {
            "message": "Clean slate completed - ready for fresh training",
            "reset_summary": reset_count,
            "total_files_removed": total_removed,
            "status": "success",
            "next_steps": [
                "Record new audio clips (30+ clips recommended)",
                "Analyze dataset for quality validation", 
                "Train voice model with clean data",
                "Test voice synthesis"
            ]
        }
        
    except Exception as e:
        logger.error(f"Reset training data error: {e}")
        return {"error": f"Reset failed: {str(e)}", "status": "failed"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Clone API v2.0 - Memory-optimized for M1 MacBook",
        "docs": "/docs",
        "health": "/healthz",
        "features": [
            "Audio upload and processing",
            "Dataset analysis with Whisper-tiny",
            "XTTS voice model training",
            "Real voice synthesis (trained models only)",
            "Memory-optimized for 8GB RAM"
        ]
    }

# Background task for periodic cleanup
@app.on_event("startup")
async def startup_tasks():
    """Initialize background tasks"""
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    """Periodic cleanup to manage memory usage"""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            
            if memory_used_gb > 5.5:  # If using >5.5GB, cleanup
                logger.info("High memory usage detected, running cleanup...")
                voice_synthesizer.cleanup()
                
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )