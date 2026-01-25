#!/bin/bash

# Stop execution if any command fails
set -e

echo "========================================"
echo "🧠 PongGame Backend Server"
echo "========================================"
echo ""
echo "🔄 Activating Python virtual environment..."
source server_env/bin/activate
echo "✅ Virtual environment activated."
echo ""

echo "📦 Installing Python dependencies..."
echo "⚙️  Running: pip install -r requirements.txt"
pip install -r requirements.txt
echo "✅ All dependencies are up to date."
echo ""

echo "🚀 Starting backend server..."
echo "⚡ Running: python run.py"
python run.py

echo ""
echo "🟢 Server is now running!"
echo "📡 Waiting for incoming connections..."
echo "💻 Happy coding!"
echo "========================================"
