#!/bin/bash

# Setup script for running SAM2 on EC2
# This script installs all necessary dependencies and downloads the SAM2 model

echo "🚀 Setting up SAM2 environment on EC2..."

# Update system packages
sudo yum update -y

# Install system dependencies
sudo yum install -y python3 python3-pip git wget unzip

# Install CUDA dependencies (for GPU instances)

    # Install CUDA toolkit if not already installed
    # Note: This might need to be adjusted based on your AMI
sudo yum install -y cuda-toolkit


# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional SAM2 dependencies
echo "📦 Installing SAM2 dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm
pip install einops
pip install transformers
pip install accelerate

# Clone SAM2 repository
echo "📥 Cloning SAM2 repository..."
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# Install SAM2
echo "📦 Installing SAM2..."
pip install -e .

# Create directories for SAM2 models
mkdir -p checkpoints
mkdir -p configs/sam2.1

# Download SAM2.1 Hiera Tiny model (smaller, faster)
echo "📥 Downloading SAM2.1 Hiera Tiny model..."
wget -O checkpoints/sam2.1_hiera_tiny.pt "https://dl.fbaipublicfiles.com/sam2/sam2.1_hiera_tiny.pt"

# Download SAM2 config file
echo "📥 Downloading SAM2 config..."
wget -O configs/sam2.1/sam2.1_hiera_t.yaml "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_t.yaml"

# Go back to project directory
cd /home/ec2-user/aws/backend

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Set up your .env file with AWS credentials"
echo "2. Test SAM2 installation: python3 -c \"import torch; from sam2.build_sam import build_sam2; print('SAM2 ready!')\""
echo "3. Run: source venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "🔧 Troubleshooting:"
echo "- If CUDA issues: nvidia-smi to check GPU"
echo "- If memory issues: Use larger instance type"
echo "- If import errors: Check SAM2 installation in /opt/dlami/nvme/sam2" 