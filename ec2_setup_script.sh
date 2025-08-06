#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Update system
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y git wget unzip cmake build-essential libopencv-dev \
                    libatlas-base-dev libboost-all-dev libprotobuf-dev protobuf-compiler \
                    libgoogle-glog-dev libgflags-dev libhdf5-dev \
                    python3-dev python3-pip libopenblas-dev \
                    libssl-dev libcurl4-openssl-dev ffmpeg

# Install Python dependencies
print_status "Installing Python dependencies..."
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python matplotlib fastapi uvicorn tqdm numpy pillow

# Create persistent directories on EBS volume
print_status "Creating persistent directories..."
mkdir -p ~/models/sam2
mkdir -p ~/models/openpose
cd ~/models

# Install SAM2
print_status "Installing SAM2..."
cd ~/models/sam2
git clone https://github.com/facebookresearch/sam2.git .
pip install -e .

# Download SAM2 checkpoint
print_status "Downloading SAM2 checkpoint..."
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/sam2/sam2.1_hiera_tiny.pt

# Create config directory structure
print_status "Setting up SAM2 config..."
mkdir -p configs/sam2.1
wget -O configs/sam2.1/sam2.1_hiera_t.yaml https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_t.yaml

# Test SAM2 installation
print_status "Testing SAM2 installation..."
python3 -c "from sam2.modeling import Sam2; print('SAM2 imported successfully')" || {
    print_error "SAM2 installation failed"
    exit 1
}

print_success "SAM2 installed successfully!"

# Install OpenPose
print_status "Installing OpenPose..."
cd ~/models/openpose
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

# Build OpenPose
print_status "Building OpenPose (this may take 30-60 minutes)..."
mkdir build && cd build
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=OFF ..
make -j$(nproc)

# Add OpenPose to PYTHONPATH
echo 'export PYTHONPATH=$PYTHONPATH:~/models/openpose/build/python' >> ~/.bashrc
source ~/.bashrc

# Test OpenPose installation
print_status "Testing OpenPose installation..."
cd ~/models/openpose/build/python
python3 -c "import openpose; print('OpenPose imported successfully')" || {
    print_warning "OpenPose Python import failed, but binary should still work"
}

print_success "OpenPose built successfully!"

# Create symbolic links for easier access
print_status "Creating symbolic links..."
sudo ln -sf ~/models/sam2 /opt/sam2
sudo ln -sf ~/models/openpose /opt/openpose

# Set up environment variables
cat >> ~/.bashrc << 'ENV_EOF'
# SAM2 and OpenPose environment
export SAM2_ROOT=/opt/sam2
export OPENPOSE_ROOT=/opt/openpose
export PYTHONPATH=$PYTHONPATH:/opt/sam2:/opt/openpose/build/python
export PATH=$PATH:/opt/openpose/build/examples/openpose
ENV_EOF

source ~/.bashrc

# Create a test script
cat > ~/test_installation.py << 'TEST_EOF'
#!/usr/bin/env python3
import sys
import os

print("Testing SAM2 and OpenPose installation...")

# Test SAM2
try:
    sys.path.append("/opt/sam2")
    from sam2.modeling import Sam2
    print("✅ SAM2 imported successfully")
except Exception as e:
    print(f"❌ SAM2 import failed: {e}")

# Test OpenPose
try:
    sys.path.append("/opt/openpose/build/python")
    import openpose
    print("✅ OpenPose imported successfully")
except Exception as e:
    print(f"❌ OpenPose import failed: {e}")

# Test OpenPose binary
openpose_bin = "/opt/openpose/build/examples/openpose/openpose.bin"
if os.path.exists(openpose_bin):
    print("✅ OpenPose binary found")
else:
    print("❌ OpenPose binary not found")

print("Installation test complete!")
TEST_EOF

chmod +x ~/test_installation.py

print_success "Installation complete!"
print_status "Running installation test..."
python3 ~/test_installation.py

print_status "Setup Summary:"
echo "📁 SAM2 installed at: ~/models/sam2"
echo "📁 OpenPose installed at: ~/models/openpose"
echo "🔗 Symbolic links created at: /opt/sam2 and /opt/openpose"
echo "🧪 Test script created at: ~/test_installation.py"
echo "💾 All installations are on EBS volume and will persist after stop/start"

