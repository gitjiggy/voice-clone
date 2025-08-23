#!/bin/bash

# Voice Clone Backend Setup Script
# Optimized for M1 MacBook with 8GB RAM

set -e

PROJECT_ROOT="/Users/nirajdesai/Documents/AI/voice-clone"
BACKEND_DIR="$PROJECT_ROOT/backend"

echo "🐍 Setting up Python 3.11 backend..."

# Navigate to backend directory
cd "$BACKEND_DIR"

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found. Please install Python 3.11 first."
    echo "   You can install it via: brew install python@3.11"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python 3.11 virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "✅ Python version: $PYTHON_VERSION"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies with memory-optimized flags
echo "📚 Installing lightweight dependencies..."
pip install --no-cache-dir -r requirements.txt

# Verify key packages
echo "🔍 Verifying installations..."
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo "✅ Backend setup complete!"
echo "🚀 To start the backend:"
echo "   cd $BACKEND_DIR"
echo "   source venv/bin/activate"
echo "   python main.py"
