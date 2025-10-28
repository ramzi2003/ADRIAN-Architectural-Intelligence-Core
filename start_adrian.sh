#!/bin/bash
# ADRIAN - Complete Startup Script
# Starts Redis, IO Service, and all dependencies in one command

echo "============================================================"
echo "🚀 Starting ADRIAN - Complete System"
echo "============================================================"

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=30
    local attempt=0
    
    echo "⏳ Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -lt $max_attempts ]; do
        if check_port $port; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start on port $port"
    return 1
}

# Start Redis if not running
if ! check_port 6379; then
    echo "📦 Starting Redis server..."
    
    # Try to start Redis
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes --port 6379
    elif command -v redis >/dev/null 2>&1; then
        redis-server --daemonize yes --port 6379
    else
        echo "❌ Redis not found. Installing Redis..."
        sudo apt update -qq
        sudo apt install -y redis-server
        sudo service redis-server start
    fi
    
    # Wait for Redis to be ready
    if ! wait_for_service 6379 "Redis"; then
        echo "❌ Failed to start Redis"
        exit 1
    fi
else
    echo "✅ Redis already running on port 6379"
fi

# Check if we're in the right directory
if [ ! -f "services/io_service/main.py" ]; then
    echo "❌ Please run this script from the ADRIAN project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected. Activating venv-linux..."
    if [ -d "venv-linux" ]; then
        source venv-linux/bin/activate
    else
        echo "❌ Virtual environment not found. Please create venv-linux first"
        exit 1
    fi
fi

echo ""
echo "🎙️ Starting ADRIAN IO Service with TTS..."
echo "   - Voice Input: Hotword detection (say 'ADRIAN')"
echo "   - Speech-to-Text: Whisper"
echo "   - Text-to-Speech: JARVIS voice (p326 + effects)"
echo "   - Audio Output: Windows speakers via WSL"
echo ""
echo "📡 Service will be available at: http://localhost:8001"
echo "🔊 Say 'ADRIAN' to start voice interaction!"
echo ""
echo "Press Ctrl+C to stop all services"
echo "============================================================"

# Start the IO Service
python services/io_service/main.py

# Cleanup on exit
echo ""
echo "🛑 Shutting down ADRIAN..."
if check_port 6379; then
    echo "📦 Stopping Redis..."
    redis-cli shutdown 2>/dev/null || sudo service redis-server stop 2>/dev/null || true
fi
echo "👋 ADRIAN shutdown complete"
