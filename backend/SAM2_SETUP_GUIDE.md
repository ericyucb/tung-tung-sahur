# Complete SAM2 Setup Guide for EC2

This guide will walk you through setting up SAM2 on your EC2 instance step by step.

## Prerequisites

### 1. EC2 Instance Requirements

**Recommended GPU Instance:**
- **Type:** `g4dn.xlarge` or larger
- **GPU:** NVIDIA T4 or better
- **RAM:** 16+ GB
- **Storage:** 50+ GB SSD
- **OS:** Amazon Linux 2 or Ubuntu 20.04+

**Alternative CPU Instance:**
- **Type:** `c5.2xlarge` or larger (slower but cheaper)
- **RAM:** 16+ GB
- **Storage:** 50+ GB SSD

### 2. Security Group Configuration

Make sure your EC2 security group allows:
- **SSH (port 22)** - for connecting
- **Custom TCP (port 8000)** - for FastAPI backend

## Step-by-Step Setup

### Step 1: Connect to Your EC2 Instance

```bash
ssh -i your-key.pem ec2-user@your-instance-ip
```

### Step 2: Navigate to Backend Directory

```bash
# Navigate to backend directory
cd /home/ec2-user/aws/backend
```

### Step 3: Run the Setup Script

```bash
# Make the script executable
chmod +x setup_ec2.sh

# Run the setup script
./setup_ec2.sh
```

This script will:
- Update system packages
- Install Python and system dependencies
- Install CUDA (if GPU detected)
- Create a virtual environment in `/backend/venv/`
- Install all Python dependencies
- Clone SAM2 repository to `/backend/sam2/`
- Download SAM2 model files

### Step 4: Configure Environment Variables

Create a `.env` file in the backend directory:

```bash
cat > .env << EOF
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_S3_BUCKET_NAME=your_bucket_name_here
AWS_REGION=us-east-1
EOF
```

### Step 5: Test SAM2 Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Test SAM2 installation
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

try:
    from sam2.build_sam import build_sam2
    print('✅ SAM2 imported successfully!')
except ImportError as e:
    print(f'❌ SAM2 import failed: {e}')
"
```

### Step 6: Test Video Processing

Create a simple test to verify everything works:

```bash
# Create a test video
python3 -c "
import cv2
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/tmp/test.mp4', fourcc, 20.0, (640,480))
for i in range(30):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out.write(frame)
out.release()
print('Test video created')
"

# Test frame extraction
python3 inference.py --input /tmp/test.mp4 --output /tmp/test_frame.jpg
```

### Step 7: Start the Backend Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Directory Structure After Setup

```
/home/ec2-user/aws/backend/
├── venv/                      # Python virtual environment
├── sam2/                      # SAM2 repository
│   ├── checkpoints/
│   │   └── sam2.1_hiera_tiny.pt
│   └── configs/
│       └── sam2.1/
│           └── sam2.1_hiera_t.yaml
├── main.py                    # FastAPI server
├── inference.py               # SAM2 inference script
├── requirements.txt           # Dependencies
├── .env                      # Environment variables
└── setup_ec2.sh             # Setup script
```

## Verification Steps

### 1. Check GPU (if using GPU instance)

```bash
nvidia-smi
```

You should see your GPU listed with memory usage.

### 2. Check SAM2 Installation

```bash
source venv/bin/activate
python3 -c "
import sys
sys.path.append('./sam2')
from sam2.build_sam import build_sam2
print('SAM2 ready!')
"
```

### 3. Test API Endpoints

```bash
# Test the root endpoint
curl http://localhost:8000/

# Test upload endpoint (replace with actual video file)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/video.mp4"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Not Available
```bash
# Check if CUDA is installed
nvcc --version

# Check if PyTorch sees CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Solution:** Make sure you're using a GPU instance and CUDA is properly installed.

#### 2. Out of Memory
```bash
# Check memory usage
free -h

# Check GPU memory
nvidia-smi
```

**Solution:** Use a larger instance type or process smaller videos.

#### 3. SAM2 Import Errors
```bash
# Check if SAM2 is installed
ls -la sam2/

# Reinstall SAM2
cd sam2
pip install -e .
```

#### 4. Model Files Missing
```bash
# Check if model files exist
ls -la sam2/checkpoints/
ls -la sam2/configs/sam2.1/
```

**Solution:** Re-run the model download commands from the setup script.

#### 5. Space Issues
```bash
# Check disk usage
df -h

# Clean up space
sudo yum clean all
pip cache purge
sudo rm -rf /tmp/*
```

**Solution:** The setup script includes `--no-cache-dir` flags to prevent space issues.

### Performance Optimization

#### 1. GPU Memory Management
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Clear GPU cache if needed
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Instance Scaling
- **For small videos:** `g4dn.xlarge` is sufficient
- **For large videos:** Use `g4dn.2xlarge` or larger
- **For batch processing:** Consider `g4dn.4xlarge`

#### 3. Storage Optimization
```bash
# Use EBS for temporary files
sudo mkfs -t xfs /dev/xvdf
sudo mount /dev/xvdf /mnt/temp
```

## Monitoring and Logs

### 1. Application Logs
```bash
# View real-time logs
tail -f /var/log/uvicorn.log

# Or run with logging
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
```

### 2. System Monitoring
```bash
# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor network
iftop
```

### 3. GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1
```

## Security Best Practices

### 1. HTTPS Setup
```bash
# Install nginx
sudo yum install -y nginx

# Configure SSL (you'll need certificates)
sudo nano /etc/nginx/conf.d/sam2.conf
```

### 2. Firewall Configuration
```bash
# Configure firewall
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### 3. API Rate Limiting
Consider implementing rate limiting in your FastAPI application.

## Cost Optimization

### 1. Spot Instances
Use spot instances for non-critical workloads to save up to 90%.

### 2. Auto Scaling
Set up auto scaling based on CPU/GPU utilization.

### 3. Reserved Instances
Use reserved instances for predictable workloads.

## Next Steps

1. **Test with your frontend:** Connect your frontend to the EC2 backend
2. **Set up monitoring:** Implement proper logging and monitoring
3. **Optimize performance:** Fine-tune based on your specific use case
4. **Scale up:** Consider load balancing for multiple instances

## Support

If you encounter issues:
1. Check the logs: `tail -f /var/log/uvicorn.log`
2. Verify dependencies: `pip list`
3. Test SAM2: Run the verification commands above
4. Check system resources: `htop`, `nvidia-smi`

Your SAM2 setup should now be ready for video processing! 🎉 