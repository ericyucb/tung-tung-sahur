#!/bin/bash

echo "🔍 Testing connection to EC2 instance..."
echo "📍 IP: 98.86.186.120"
echo "🔑 Key: backend/sam2-key-private.pem"

# Test SSH connection
ssh -i backend/sam2-key-private.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@98.86.186.120 "echo '✅ Connection successful!'" && {
    echo "✅ EC2 instance is accessible!"
    echo "🚀 Ready to run setup script: ./setup_ec2_sam2_openpose.sh"
} || {
    echo "❌ Failed to connect to EC2 instance"
    echo "Please check:"
    echo "1. Instance is running"
    echo "2. IP address is correct: 98.86.186.120"
    echo "3. Security group allows SSH (port 22)"
    echo "4. Key file exists: backend/sam2-key-private.pem"
} 