#!/bin/bash

# Setup script for running SAM2 on EC2
# This script installs all necessary dependencies and downloads the SAM2 model
# Run this script from the /backend directory

echo "🚀 Setting up SAM2 environment on EC2..."

# Ensure we're in the backend directory
cd /aws/backend

# Update system packages
sudo yum update -y

# Install system dependencies
sudo yum install -y python3 python3-pip git wget unzip ffmpeg

# Clean up any existing space issues
echo "🧹 Cleaning up disk space..."
sudo yum clean all
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# Install CUDA dependencies (for GPU instances)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, installing CUDA dependencies..."
    # Install CUDA toolkit if not already installed
    # Note: This might need to be adjusted based on your AMI
    sudo yum install -y cuda-toolkit
else
    echo "⚠️  No GPU detected, will use CPU-only mode"
fi

# Create virtual environment in backend directory
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies in smaller chunks
echo "📦 Installing Python dependencies..."
pip install fastapi uvicorn python-multipart boto3 python-dotenv --no-cache-dir
pip install opencv-python matplotlib pillow numpy --no-cache-dir

# Install PyTorch (CPU version first, then GPU if needed)
echo "📦 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
else
    echo "💻 Installing PyTorch CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
fi

# Install additional SAM2 dependencies
echo "📦 Installing SAM2 dependencies..."
pip install timm --no-cache-dir
pip install einops --no-cache-dir
pip install transformers --no-cache-dir
pip install accelerate --no-cache-dir

# Clone SAM2 repository in backend directory
echo "📥 Cloning SAM2 repository..."
if [ -d "sam2" ]; then
    echo "🗑️  Removing existing sam2 directory..."
    rm -rf sam2
fi

git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Install SAM2
echo "📦 Installing SAM2..."
pip install -e . --no-cache-dir

# Create directories for SAM2 models
mkdir -p checkpoints
mkdir -p configs/sam2.1

# Download SAM2.1 Hiera Tiny model (smaller, faster)
echo "📥 Downloading SAM2.1 Hiera Tiny model..."
if [ ! -f "checkpoints/sam2.1_hiera_tiny.pt" ]; then
    wget -O checkpoints/sam2.1_hiera_tiny.pt "https://dl.fbaipublicfiles.com/sam2/sam2.1_hiera_tiny.pt"
else
    echo "✅ SAM2 model already exists"
fi

# Download SAM2 config file
echo "📥 Downloading SAM2 config..."
if [ ! -f "configs/sam2.1/sam2.1_hiera_t.yaml" ]; then
    wget -O configs/sam2.1/sam2.1_hiera_t.yaml "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_t.yaml"
else
    echo "✅ SAM2 config already exists"
fi

# Go back to backend directory
cd /home/ec2-user/aws/backend

# Clean up pip cache to save space
echo "🧹 Cleaning up pip cache..."
pip cache purge

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up your .env file with AWS credentials"
echo "2. Test SAM2 installation: source venv/bin/activate && python3 -c \"import torch; from sam2.build_sam import build_sam2; print('SAM2 ready!')\""
echo "3. Run: source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "🔧 Troubleshooting:"
echo "- If CUDA issues: nvidia-smi to check GPU"
echo "- If memory issues: Use larger instance type"
echo "- If import errors: Check SAM2 installation in sam2 directory"
echo "- If space issues: df -h to check disk usage" 