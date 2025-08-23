# Voice Clone - AI-Powered Voice Synthesis Platform

A memory-optimized voice cloning platform designed for M1 MacBook with 8GB RAM. This monorepo contains both backend (Python 3.11) and frontend (Node 20) components with strict local-only operation.

## 🏗️ Architecture

```
voice-clone/
├── backend/           # FastAPI backend (Python 3.11)
├── frontend/          # React frontend (Node 20)
├── data/
│   ├── raw/          # Original voice recordings
│   └── processed/    # Preprocessed audio data
├── models/
│   └── voice_clone/  # Trained voice models
├── outputs/          # Generated voice samples
├── logs/             # Application logs
└── scripts/          # Setup and utility scripts
```

## 🔧 Prerequisites

- **macOS** (M1 MacBook optimized)
- **Python 3.11** (install via `brew install python@3.11`)
- **Node 20** via **nvm** (`curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash`)
- **8GB RAM** (system uses max 6GB, leaving 2GB for macOS)

## 🚀 Quick Start

### 1. Setup Backend (Python 3.11)

```bash
cd /Users/nirajdesai/Documents/AI/voice-clone
./scripts/setup_backend.sh
```

### 2. Setup Frontend (Node 20)

```bash
./scripts/setup_frontend.sh
```

### 3. Start All Services

```bash
./scripts/start_services.sh
```

## 🌐 Service Endpoints

| Service | URL | Description |
|---------|-----|-------------|
| **Backend API** | http://127.0.0.1:8000 | FastAPI REST API |
| **Frontend** | http://127.0.0.1:4001 | React web interface |
| **API Documentation** | http://127.0.0.1:8000/docs | Interactive Swagger docs |
| **Health Check** | http://127.0.0.1:8000/healthz | System status endpoint |

## 🔍 Manual Startup Commands

### Backend
```bash
cd /Users/nirajdesai/Documents/AI/voice-clone/backend
source venv/bin/activate
python main.py
```

### Frontend
```bash
cd /Users/nirajdesai/Documents/AI/voice-clone/frontend
nvm use 20
export VITE_API_BASE=http://127.0.0.1:8000
npm run dev
```

## 🩺 Health Monitoring

The system includes comprehensive health monitoring:

### Backend Health Check (`GET /healthz`)
```json
{
  "python_version": "3.11.x",
  "torch_version": "2.1.0",
  "memory_available_gb": 4.23,
  "disk_free_gb": 85.7,
  "model_loaded": false,
  "status": "healthy",
  "platform": "macOS-14.6-arm64-arm-64bit"
}
```

### Frontend System Status Widget
- **Real-time monitoring** of backend health
- **Memory usage tracking** with color-coded alerts
- **Automatic refresh** every 10 seconds
- **Visual status indicators** for all system components

## 🎙️ Voice Recording Guide

### Optimized Recording Prompts
Use the provided prompts in `/scripts/recording_prompts.txt`:

```bash
cat /Users/nirajdesai/Documents/AI/voice-clone/scripts/recording_prompts.txt
```

### Recording Setup
1. **Environment**: Quiet room, minimal background noise
2. **Microphone**: 6-8 inches distance, consistent positioning
3. **Format**: WAV files, 44.1kHz, 16-bit minimum
4. **Naming**: `prompt_01.wav`, `prompt_02.wav`, etc.
5. **Storage**: Save to `/data/raw/` directory

## 🧠 Memory Optimization

### System Constraints
- **Total RAM**: 8GB
- **Available for app**: 6GB maximum
- **Reserved for macOS**: 2GB minimum

### Optimization Features
- **CPU-only PyTorch** for M1 compatibility
- **Lightweight dependencies** with minimal memory footprint
- **Streaming audio processing** to avoid memory spikes
- **Automatic garbage collection** for long-running processes

## 🔧 Development

### Backend Development
```bash
cd backend
source venv/bin/activate
pip install pytest
pytest  # Run tests
```

### Frontend Development
```bash
cd frontend
nvm use 20
npm run test     # Run Vitest tests
npm run lint     # ESLint checking
npm run build    # Production build
```

## 📊 Package Versions

### Backend Dependencies
- **FastAPI**: 0.104.1
- **PyTorch**: 2.1.0 (CPU-only)
- **NumPy**: 1.24.4
- **Librosa**: 0.10.1
- **Whisper**: 20231117
- **TTS (Coqui)**: 0.19.6

### Frontend Dependencies
- **React**: 18.2.0
- **Vite**: 5.0.8
- **TypeScript**: 5.2.2
- **Tailwind CSS**: 3.3.6
- **Vitest**: 1.0.4

## ✅ Self-Verification Checklist

Run these commands to verify proper setup:

### 1. Python Version Check
```bash
cd /Users/nirajdesai/Documents/AI/voice-clone/backend
source venv/bin/activate
python --version
# Expected: Python 3.11.x
```

### 2. Node Version Check
```bash
nvm use 20
node -v
# Expected: v20.x.x
```

### 3. Backend Accessibility
```bash
curl http://127.0.0.1:8000/healthz
# Expected: JSON response with system info
```

### 4. API Documentation
Open: http://127.0.0.1:8000/docs
- Should display interactive Swagger documentation

### 5. Frontend System Status
Open: http://127.0.0.1:4001
- Should display "Voice Clone" interface
- System Status widget should show green indicators
- Memory usage should be displayed

## 🚨 Troubleshooting

### Port Conflicts
```bash
# Check port usage
lsof -ti:8000  # Backend port
lsof -ti:4001  # Frontend port

# Kill processes if needed
kill $(lsof -ti:8000)
kill $(lsof -ti:4001)
```

### Memory Issues
- Monitor via System Status widget
- Restart services if memory usage exceeds 5GB
- Check for memory leaks in logs

### Setup Failures
- Ensure Python 3.11 is installed: `brew install python@3.11`
- Verify nvm installation: `nvm --version`
- Check disk space: minimum 10GB free required

## 📝 Project Status

- ✅ **Monorepo structure** created
- ✅ **Backend** (Python 3.11, FastAPI) configured
- ✅ **Frontend** (Node 20, React, TypeScript) configured
- ✅ **Health monitoring** implemented
- ✅ **Recording prompts** provided
- ✅ **Memory optimization** configured
- ✅ **Documentation** complete

## 🎯 **4-Hour Hackathon Timeline**

### **Phase 1: Setup (30 minutes)**
```bash
# 1. Environment Setup (15 minutes)
./scripts/setup_backend.sh    # Python 3.11 + dependencies
./scripts/setup_frontend.sh   # Node 20 + React/TypeScript

# 2. Service Start (5 minutes)
./scripts/start_services.sh   # Backend + Frontend

# 3. Verification (10 minutes)
curl http://127.0.0.1:8000/healthz    # Backend health
open http://127.0.0.1:4001             # Frontend UI
```

### **Phase 2: Recording (45 minutes)**
```
Target: 30 clips × 30-45 seconds = 15-22.5 minutes audio
Quality Gate: Green status (>15min) for optimal training
Process: Guided prompts → Real-time feedback → Auto-validation
```

### **Phase 3: Training (60-120 minutes)**
```
Analysis: Whisper-tiny transcription + quality scoring
Training: XTTS v2 fine-tuning with progress monitoring
Output: YOUR actual voice model (no synthetic fallbacks)
```

### **Phase 4: Demo (30 minutes)**
```
Synthesis: Custom text → Your cloned voice
Validation: Side-by-side original vs AI comparison
Evidence: Generated audio files in /outputs/
```

## 🏆 **Success Metrics**

- ✅ **Recognizable voice in under 4 hours**
- ✅ **Quality commitment: YOUR actual voice cloned**
- ✅ **Memory efficiency: Works on M1 MacBook 8GB RAM**
- ✅ **Evidence files: Original vs cloned audio comparison**

## 🚀 **Ready for Hackathon Demo**

**All services verified working and ready for demonstration!**

---

**Real voice cloning - M1 optimized - Hackathon ready**
