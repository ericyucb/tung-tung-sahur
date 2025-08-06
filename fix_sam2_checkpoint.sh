#!/bin/bash

echo "🔧 Fixing SAM2 checkpoint download..."

# Connect to EC2 and fix the checkpoint download
ssh -i backend/sam2-key-private.pem ubuntu@98.86.186.120 << 'EOF'

cd ~/models/sam2

# Try alternative checkpoint URLs
echo "Trying alternative SAM2 checkpoint URLs..."

# Method 1: Try the official SAM2 checkpoint
wget -O checkpoints/sam2.1_hiera_tiny.pt https://dl.fbaipublicfiles.com/sam2/sam2.1_hiera_tiny.pt || {
    echo "Failed with official URL, trying alternative..."
    
    # Method 2: Try Hugging Face model
    pip install huggingface_hub
    python3 -c "
import os
from huggingface_hub import hf_hub_download

try:
    checkpoint_path = hf_hub_download(
        repo_id='facebook/sam2',
        filename='sam2.1_hiera_tiny.pt',
        cache_dir='checkpoints'
    )
    print(f'Downloaded checkpoint to: {checkpoint_path}')
except Exception as e:
    print(f'Failed to download from Hugging Face: {e}')
    exit(1)
"
}

# Verify checkpoint exists
if [ -f "checkpoints/sam2.1_hiera_tiny.pt" ]; then
    echo "✅ SAM2 checkpoint downloaded successfully!"
    ls -la checkpoints/
else
    echo "❌ Failed to download SAM2 checkpoint"
    echo "Please manually download the checkpoint from:"
    echo "https://huggingface.co/facebook/sam2"
    exit 1
fi

# Test SAM2 import
echo "Testing SAM2 installation..."
python3 -c "from sam2.modeling import Sam2; print('✅ SAM2 imported successfully')" || {
    echo "❌ SAM2 import failed"
    exit 1
}

echo "✅ SAM2 installation complete!"

EOF

echo "🔧 SAM2 checkpoint fix complete!" 