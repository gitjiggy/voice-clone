#!/bin/bash

# Voice Clone Frontend Setup Script
# Node 20 via nvm only

set -e

PROJECT_ROOT="/Users/nirajdesai/Documents/AI/voice-clone"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

echo "🟢 Setting up Node 20 frontend..."

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# Check if nvm is available after loading
if ! command -v nvm &> /dev/null; then
    echo "❌ nvm not found. Please install nvm first:"
    echo "   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "   Then restart your terminal and run this script again."
    exit 1
fi

# Install and use Node 20
echo "📦 Installing Node 20..."
nvm install 20
nvm use 20

# Verify Node version
NODE_VERSION=$(node -v)
echo "✅ Node version: $NODE_VERSION"

# Navigate to frontend directory
cd "$FRONTEND_DIR"

# Install dependencies
echo "📚 Installing frontend dependencies..."
npm install

# Verify key packages
echo "🔍 Verifying installations..."
npm list vite --depth=0
npm list react --depth=0
npm list typescript --depth=0

echo "✅ Frontend setup complete!"
echo "🚀 To start the frontend:"
echo "   cd $FRONTEND_DIR"
echo "   nvm use 20"
echo "   npm run dev"
