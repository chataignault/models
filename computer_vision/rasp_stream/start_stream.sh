#!/bin/bash
# Helper script to start the video stream

# Configuration
RPI_HOST="${RPI_HOST:-raspberrypi.local}"
RPI_USER="${RPI_USER:-pi}"
PORT="${PORT:-8888}"
VERBOSE="${VERBOSE:-0}"

echo "Raspberry Pi Camera Streaming Setup"
echo "===================================="
echo "RPi Host: $RPI_HOST"
echo "RPi User: $RPI_USER"
echo "Port: $PORT"
echo ""

# Check if server script exists
if [ ! -f "laptop_client.py" ]; then
    echo "Error: laptop_client.py not found in current directory"
    exit 1
fi

echo "Step 1: Checking if SSH tunnel already exists..."
if pgrep -f "ssh -L ${PORT}:localhost:${PORT}" > /dev/null; then
    echo "SSH tunnel already running. Killing old tunnel..."
    pkill -f "ssh -L ${PORT}:localhost:${PORT}"
    sleep 1
fi

echo "Step 2: Creating SSH tunnel..."
ssh -L ${PORT}:localhost:${PORT} -N -f ${RPI_USER}@${RPI_HOST}

if [ $? -eq 0 ]; then
    echo "SSH tunnel created successfully"
else
    echo "Error creating SSH tunnel"
    echo "Please check:"
    echo "  1. Can you SSH to the RPi? Try: ssh ${RPI_USER}@${RPI_HOST}"
    echo "  2. Is the server running on the RPi? Try: ssh ${RPI_USER}@${RPI_HOST} 'pgrep -f rpi_server.py'"
    exit 1
fi

echo ""
echo "Step 3: Waiting for tunnel to establish..."
sleep 2

echo ""
echo "Step 4: Checking if server is reachable..."
if timeout 2 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
    echo "Server is reachable on localhost:${PORT}"
else
    echo "Warning: Cannot connect to localhost:${PORT}"
    echo "Make sure rpi_server.py is running on the Raspberry Pi"
fi

echo ""
echo "Step 5: Starting video client..."
echo "Press 'q' in the video window to quit"
echo ""

sleep 1

if [ "$VERBOSE" = "1" ]; then
    uv run laptop_client.py --port ${PORT} --verbose
else
    uv run laptop_client.py --port ${PORT}
fi

echo ""
echo "Cleaning up..."
# Kill the SSH tunnel
pkill -f "ssh -L ${PORT}:localhost:${PORT}"

echo "Done!"
