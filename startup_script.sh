#!/bin/bash

# Startup script for SAM2 AWS Processing Backend
# This script will be run when the EC2 instance starts

echo "🚀 Starting SAM2 AWS Processing Backend..."

# Set up environment
export PATH="/home/ubuntu/.local/bin:$PATH"

# Navigate to backend directory
cd /opt/backend

# Start the server
echo "Starting uvicorn server..."
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > /opt/backend/server.log 2>&1 &

echo "✅ Server started successfully!"
echo "📡 Backend available at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
echo "📋 Check logs with: tail -f /opt/backend/server.log" 