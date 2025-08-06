# 🚀 SAM2 & OpenPose EC2 Setup Guide

## 📍 Instance Details
- **IP Address**: `98.86.186.120`
- **Key File**: `backend/sam2-key-private.pem`
- **User**: `ubuntu`

## 🔧 Quick Setup

### 1. Test Connection
```bash
./test_connection.sh
```

### 2. Run Full Setup
```bash
./setup_ec2_sam2_openpose.sh
```

## 📋 Manual Setup Steps

If you prefer to run commands manually:

### 1. Connect to EC2
```bash
ssh -i backend/sam2-key-private.pem ubuntu@98.86.186.120
```

### 2. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 3. Install Dependencies
```bash
sudo apt install -y git wget unzip cmake build-essential libopencv-dev \
                    libatlas-base-dev libboost-all-dev libprotobuf-dev protobuf-compiler \
                    libgoogle-glog-dev libgflags-dev libhdf5-dev \
                    python3-dev python3-pip libopenblas-dev \
                    libssl-dev libcurl4-openssl-dev ffmpeg
```

### 4. Install Python Dependencies
```bash
python3 -m pip install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python matplotlib fastapi uvicorn tqdm numpy pillow
```

### 5. Install SAM2
```bash
mkdir -p ~/models/sam2
cd ~/models/sam2
git clone https://github.com/facebookresearch/sam2.git .
pip install -e .

# Download checkpoint
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/sam2/sam2.1_hiera_tiny.pt

# Setup config
mkdir -p configs/sam2.1
wget -O configs/sam2.1/sam2.1_hiera_t.yaml https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_t.yaml
```

### 6. Install OpenPose
```bash
mkdir -p ~/models/openpose
cd ~/models/openpose
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .

# Build OpenPose (takes 30-60 minutes)
mkdir build && cd build
cmake -DBUILD_PYTHON=ON -DBUILD_EXAMPLES=OFF ..
make -j$(nproc)
```

### 7. Setup Environment
```bash
# Add to ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:~/models/openpose/build/python' >> ~/.bashrc
source ~/.bashrc

# Create symbolic links
sudo ln -sf ~/models/sam2 /opt/sam2
sudo ln -sf ~/models/openpose /opt/openpose
```

## 🧪 Testing Installation

### Test SAM2
```bash
python3 -c "from sam2.modeling import Sam2; print('SAM2 imported successfully')"
```

### Test OpenPose
```bash
cd ~/models/openpose/build/python
python3 -c "import openpose; print('OpenPose imported successfully')"
```

### Test OpenPose Binary
```bash
ls -la /opt/openpose/build/examples/openpose/openpose.bin
```

## 📁 Directory Structure

After installation:
```
~/models/
├── sam2/                    # SAM2 installation
│   ├── checkpoints/
│   │   └── sam2.1_hiera_tiny.pt
│   └── configs/
│       └── sam2.1/
│           └── sam2.1_hiera_t.yaml
└── openpose/               # OpenPose installation
    └── build/
        ├── examples/
        │   └── openpose/
        │       └── openpose.bin
        └── python/
```

## 🔗 Symbolic Links

- `/opt/sam2` → `~/models/sam2`
- `/opt/openpose` → `~/models/openpose`

## 💾 Persistence

All installations are on the **EBS volume** and will persist after:
- Instance stop/start
- Reboot
- System updates

## 🚀 Running Your Code

### Copy Backend Files
```bash
scp -i backend/sam2-key-private.pem -r backend/ ubuntu@98.86.186.120:~/
```

### Run Inference
```bash
ssh -i backend/sam2-key-private.pem ubuntu@98.86.186.120
cd ~/backend
python3 inference.py --video input.mp4 --points '[[100,100]]' --video_output output.mp4
```

## 🔧 Troubleshooting

### Connection Issues
1. Check instance is running
2. Verify IP address: `98.86.186.120`
3. Ensure security group allows SSH (port 22)
4. Check key file permissions: `chmod 400 backend/sam2-key-private.pem`

### Installation Issues
1. **SAM2 import error**: Check CUDA installation
2. **OpenPose build fails**: Ensure enough disk space (100GB+)
3. **Memory issues**: Use `g4dn.xlarge` or larger instance

### Common Commands
```bash
# Check GPU
nvidia-smi

# Check disk space
df -h

# Check memory
free -h

# Check CUDA
nvcc --version
```

## 📞 Support

If you encounter issues:
1. Check the logs in the setup script
2. Verify all dependencies are installed
3. Ensure sufficient disk space and memory
4. Test individual components separately

---

**Ready to run**: `./setup_ec2_sam2_openpose.sh` 