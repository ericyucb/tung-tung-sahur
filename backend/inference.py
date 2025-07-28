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

# SAM2 imports
from sam2.build_sam import build_sam2_video_predictor

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from video {video_path}")
            return False
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save the frame
        cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        cap.release()
        
        print(f"First frame extracted and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return False

def create_sam2_segmentation(frame_path, points, output_path):
    """Create segmentation using the real SAM2 model."""
    try:
        # Initialize SAM2 predictor
        cfg = "configs/sam2/sam2_hiera_s.yaml"
        ckpt = "checkpoints/sam2.1_hiera_small.pt"
        
        # Check if we're in the SAM2 directory, if not, use absolute paths
        if not os.path.exists(cfg):
            # We're likely in the backend directory, need to go to SAM2 directory
            sam2_dir = os.path.expanduser("~/sam2")
            cfg = os.path.join(sam2_dir, cfg)
            ckpt = os.path.join(sam2_dir, ckpt)
        
        print(f"Using config: {cfg}")
        print(f"Using checkpoint: {ckpt}")
        
        # Build the predictor
        predictor = build_sam2_video_predictor(model_cfg=cfg, checkpoint=ckpt)
        
        # Convert points to the format expected by SAM2
        points_np = np.array(points)
        labels = np.ones(len(points), dtype=np.int32)  # All positive points
        
        print(f"Points: {points_np}")
        print(f"Labels: {labels}")
        
        # Run segmentation
        result = predictor(
            source=frame_path,
            points=points_np,
            labels=labels,
            mode="segment",
            conf=0.25
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
        if hasattr(result, 'masks') and result.masks is not None:
            mask = result.masks[0]  # Use the first mask
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
        return False

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