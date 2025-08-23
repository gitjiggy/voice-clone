#!/bin/bash

# Voice Clone Services Startup Script
# Starts both backend and frontend with proper environment

set -e

PROJECT_ROOT="/Users/nirajdesai/Documents/AI/voice-clone"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "🚀 Starting Voice Clone Services..."

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -ti:$port > /dev/null 2>&1; then
        echo "❌ Port $port is already in use"
        return 1
    fi
    return 0
}

# Check required ports
if ! check_port 8000; then
    echo "   Backend port 8000 is occupied. Please free it and try again."
    exit 1
fi

if ! check_port 4001; then
    echo "   Frontend port 4001 is occupied. Please free it and try again."
    exit 1
fi

# Start backend in background
echo "🐍 Starting backend on 127.0.0.1:8000..."
cd "$BACKEND_DIR"
if [ ! -d "venv" ]; then
    echo "❌ Backend venv not found. Run setup_backend.sh first."
    exit 1
fi

source venv/bin/activate
python main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 3

# Check if backend is running
if ! curl -s http://127.0.0.1:8000/healthz > /dev/null; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Backend running (PID: $BACKEND_PID)"

# Start frontend
echo "🟢 Starting frontend on 127.0.0.1:4001..."
cd "$FRONTEND_DIR"

# Load nvm and use Node 20
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 20

# Set environment variable and start
export VITE_API_BASE=http://127.0.0.1:8000
npm run dev &
FRONTEND_PID=$!

echo "⏳ Waiting for frontend to initialize..."
sleep 5

# Check if frontend is running
if ! curl -s http://127.0.0.1:4001 > /dev/null; then
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

echo "✅ Frontend running (PID: $FRONTEND_PID)"
echo ""
echo "🎉 Voice Clone services are running!"
echo "   Backend:  http://127.0.0.1:8000"
echo "   Frontend: http://127.0.0.1:4001"
echo "   API Docs: http://127.0.0.1:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit 0' INT
wait
