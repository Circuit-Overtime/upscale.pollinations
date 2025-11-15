#!/bin/bash

echo "Starting ESRGAN model servers..."
cd ~/upscaler.pollinations
# Function to cleanup on exit
cleanup() {
    echo "Stopping all servers..."
    pkill -f "python api/model_server.py"
    pkill -f "python api/app.py"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start 5 model servers on different ports
for port in {5002..5006}; do
    echo "Starting model server on port $port"
    python api/model_server.py $port &
    sleep 2  # Small delay between starts
done

echo "All model servers started. Waiting for them to initialize..."
sleep 10

# Check if uploads directory exists, create if not
mkdir -p uploads

echo "Starting Quart application with background cleanup..."
python api/app.py

# Wait for background processes
wait