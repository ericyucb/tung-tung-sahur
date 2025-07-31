import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import cv2
import tempfile
import shutil
import torch

# Add SAM2 to path for imports
sys.path.append("/opt/dlami/nvme/sam2")

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        # Read the first frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            cap.release()
            return False
        
        # Save the frame
        success = cv2.imwrite(output_path, frame)
        
        # Release the video capture
        cap.release()
        
        if success:
            print(f"First frame extracted and saved to: {output_path}")
            return True
        else:
            print(f"Error: Could not save frame to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return False

def create_sam2_segmentation(frame_path, points, output_path):
    """Create segmentation using the real SAM2 model."""
    try:
        # Get the SAM2 directory
        sam2_dir = os.path.expanduser("/opt/dlami/nvme/sam2")
        
        # Store current directory
        original_dir = os.getcwd()
        
        # Change directory to SAM2 root before loading the config
        os.chdir(sam2_dir)
        print(f"Changed working directory to: {os.getcwd()}")
        
        # SAM2 imports
        
        ckpt_path = 'checkpoints/sam2.1_hiera_tiny.pt'
        device = 'cuda'  # Use GPU by default
        
        print(f"Using checkpoint: {ckpt_path}")
        print(f"Using device: {device}")
        
        # Use SAM2ImagePredictor for single image segmentation
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_sam2(model_cfg, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(model)
        
        # Convert points to the format expected by SAM2
        points_np = np.array(points)
        labels = np.ones(len(points), dtype=np.int32)  # All positive points
        
        print(f"Points: {points_np}")
        print(f"Labels: {labels}")
        
        # Load image with OpenCV and set it in predictor
        import cv2
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Could not load image: {frame_path}")
        
        predictor.set_image(img)
        
        # Run segmentation
        masks, scores, logits = predictor.predict(
            point_coords=points_np,
            point_labels=labels
        )
        
        # Load the original image for visualization
        img = Image.open(frame_path)
        img_np = np.array(img)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("SAM2 Segmentation")
        ax.imshow(img_np)
        
        # Show points
        if len(points_np) > 0:
            ax.scatter(points_np[:, 0], points_np[:, 1], color='lime', marker='*', s=300, edgecolors='white')
        
        # Show mask overlay if available
        if masks is not None and len(masks) > 0:
            mask = masks[0]  # Use the first mask
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 4))
            mask_rgb[:, :, 1] = 1  # green
            mask_rgb[:, :, 3] = 0.5 * mask
            ax.imshow(mask_rgb)
        
        plt.axis("on")
        plt.grid(True)
        
        # Save the visualization
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"SAM2 segmentation saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating SAM2 segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally: # Ensure we change back to original directory
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description='SAM2 Video Processing')
    parser.add_argument('--input', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output frame file path')
    parser.add_argument('--frame', type=str, help='Input frame file path for segmentation')
    parser.add_argument('--points', type=str, help='Points for segmentation (JSON string)')
    parser.add_argument('--frame_dir', type=str, help='Frame directory for output')
    
    args = parser.parse_args()
    
    # Frame extraction mode
    if args.input and args.output:
        success = extract_first_frame(args.input, args.output)
        if not success:
            sys.exit(1)
    
    # Segmentation mode
    elif args.frame and args.points and args.frame_dir:
        points = json.loads(args.points)
        output_path = os.path.splitext(args.frame)[0] + '_segmented.png'
        success = create_sam2_segmentation(args.frame, points, output_path)
        
        if not success:
            sys.exit(1)
    
    else:
        print("Usage:")
        print("  For frame extraction: --input <video> --output <frame>")
        print("  For segmentation: --frame <frame> --points <json> --frame_dir <dir>")
        sys.exit(1)

if __name__ == "__main__":
    main()