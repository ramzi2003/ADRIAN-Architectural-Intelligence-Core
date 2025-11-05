"""
Main startup script for ADRIAN.
Starts all required services for full voice interaction.
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

# Project root
project_root = Path(__file__).parent

# Services to start
services = {
    "IO Service": {
        "script": "services/io_service/main.py",
        "port": 8001,
        "description": "Voice input, STT, hotword detection"
    },
    "Processing Service": {
        "script": "services/processing_service/main.py",
        "port": 8002,
        "description": "NLP, intent classification, response generation"
    },
    "Output Service": {
        "script": "services/output_service/main.py",
        "port": 8006,
        "description": "TTS, audio output"
    }
}

# Store process handles
processes = {}


def check_redis():
    """Check if Redis is running."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
        r.ping()
        return True
    except ImportError:
        print("❌ Redis Python library not installed. Install with: pip install redis")
        return False
    except:
        return False


def start_service(name, config):
    """Start a service in a separate process."""
    script_path = project_root / config["script"]
    
    print(f"🚀 Starting {name}... ({config['description']})")
    
    # Show logs in real-time (don't capture output)
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(project_root)
    )
    
    processes[name] = process
    print(f"   PID: {process.pid}")
    
    # Give it a moment to start
    time.sleep(2)
    
    return process


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Shutting down ADRIAN...")
    
    # Stop all services
    for name, process in processes.items():
        if process.poll() is None:  # Still running
            print(f"   Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"   ✅ {name} stopped")
    
    print("👋 ADRIAN shutdown complete")
    sys.exit(0)


def main():
    """Main startup function."""
    print("=" * 60)
    print("A.D.R.I.A.N - Advanced Digital Reasoning Intelligence Assistant Network")
    print("=" * 60)
    print()
    
    # Check Redis
    print("📡 Checking Redis connection...")
    if not check_redis():
        print("❌ Redis is not running!")
        print("   Please start Redis first:")
        print("   - Linux/WSL: sudo service redis-server start")
        print("   - Windows: redis-server")
        print("   - Docker: docker run -d -p 6379:6379 redis")
        return 1
    
    print("✅ Redis is running")
    print()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start all services
    print("🚀 Starting ADRIAN services...")
    print()
    
    for name, config in services.items():
        try:
            start_service(name, config)
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            signal_handler(None, None)
            return 1
    
    print()
    print("=" * 60)
    print("✅ ADRIAN is ready!")
    print("=" * 60)
    print()
    print("📢 Say 'ADRIAN' to start a conversation")
    print("📝 Services running:")
    for name, config in services.items():
        print(f"   • {name} (port {config['port']})")
    print()
    print("Press Ctrl+C to stop all services")
    print()
    
    # Monitor processes
    try:
        while True:
            # Check if any process died
            for name, process in list(processes.items()):
                if process.poll() is not None:  # Process ended
                    print(f"⚠️  {name} stopped unexpectedly (exit code: {process.returncode})")
                    # Restart it
                    print(f"🔄 Restarting {name}...")
                    start_service(name, services[name])
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        signal_handler(None, None)
        return 0


if __name__ == "__main__":
    sys.exit(main())

