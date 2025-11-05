#!/bin/bash
# ADRIAN WSL Startup Script
# Starts Processing and Output Services (IO Service runs separately in Windows)

echo "============================================================"
echo "🚀 Starting ADRIAN Services in WSL"
echo "   (Processing Service + Output Service)"
echo "   Note: Run IO Service separately in Windows PowerShell"
echo "============================================================"
echo ""

# Check Redis
echo "📡 Checking Redis connection..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running!"
    echo "   Please start Redis first: sudo service redis-server start"
    exit 1
fi
echo "✅ Redis is running"
echo ""

# Check if we're in the right directory
if [ ! -f "services/processing_service/main.py" ]; then
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

# Services to start (Processing + Output only)
echo "🚀 Starting Processing Service..."
python services/processing_service/main.py &
PROCESSING_PID=$!

sleep 3

echo "🚀 Starting Output Service..."
python services/output_service/main.py &
OUTPUT_PID=$!

sleep 3

echo ""
echo "============================================================="
echo "✅ WSL Services ready!"
echo "============================================================="
echo ""
echo "📝 Services running in WSL:"
echo "   • Processing Service (port 8002) - PID: $PROCESSING_PID"
echo "   • Output Service (port 8006) - PID: $OUTPUT_PID"
echo ""
echo "⚠️  Next step: Run IO Service in Windows PowerShell:"
echo "   .\\start_adrian_windows.ps1"
echo ""
echo "Press Ctrl+C to stop all WSL services"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping services...'; kill $PROCESSING_PID $OUTPUT_PID 2>/dev/null; wait; echo '✅ Stopped'; exit" INT TERM

# Keep script running
wait


