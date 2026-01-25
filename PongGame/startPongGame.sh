#!/bin/bash

# Stop execution if any command fails
set -e

echo "========================================"
echo "🚀 PongGame Development Environment"
echo "========================================"
echo ""
echo "🔍 Checking project dependencies..."
echo "📦 Installing Node modules (npm install)..."
npm install
echo "✅ Dependencies installed successfully."
echo ""

echo "🎮 Launching PongGame in development mode..."
echo "⚡ Running: npm run dev"
npm run dev

echo ""
echo "🟢 PongGame is now running!"
echo "🌐 Open your browser and enjoy the game."
echo "💻 Happy coding!"
echo "========================================"
