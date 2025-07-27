import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import cv2

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
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
        cv2.imwrite(output_path, frame)
        
        # Release the video capture object
        cap.release()
        
        print(f"First frame extracted and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return False

def create_simple_segmentation(frame_path, points, output_path):
    """Create a simple segmentation visualization around the clicked points."""
    try:
        # Load the image
        img = Image.open(frame_path)
        img_np = np.array(img)
        
        # Create a mask based on the points
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        
        # For each point, create a circular region around it
        for point in points:
            x, y = int(point[0]), int(point[1])
            # Create a circle around the point
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            circle_mask = dist_from_center <= 50  # 50 pixel radius
            mask = mask | circle_mask
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Simple Segmentation Preview (Mock SAM2)")
        ax.imshow(img_np)
        
        # Show points
        points_np = np.array(points)
        if len(points_np) > 0:
            ax.scatter(points_np[:, 0], points_np[:, 1], color='green', marker='*', s=300, edgecolor='white', linewidth=1.25)
        
        # Show mask overlay
        mask_rgb = np.zeros((h, w, 4))
        mask_rgb[:, :, 1] = 1  # green
        mask_rgb[:, :, 3] = 0.5 * mask
        ax.imshow(mask_rgb)
        
        plt.axis("on")
        plt.grid(True)
        
        # Save the visualization
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Simple segmentation saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating segmentation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract first frame or run simple segmentation on a frame with user points.")
    parser.add_argument('--input', type=str, help='Input video file path (for frame extraction)')
    parser.add_argument('--output', type=str, help='Output image file path (for frame extraction)')
    parser.add_argument('--frame', type=str, help='Path to the frame image (jpg/png) for segmentation')
    parser.add_argument('--points', type=str, help='JSON string or file path with points (list of [x, y])')
    parser.add_argument('--frame_dir', type=str, help='Directory containing the frame')
    args = parser.parse_args()

    # Handle frame extraction
    if args.input and args.output:
        success = extract_first_frame(args.input, args.output)
        if not success:
            sys.exit(1)
        return
    
    # Handle segmentation
    if args.frame and args.points and args.frame_dir:
        # Load points
        if os.path.isfile(args.points):
            with open(args.points, 'r') as f:
                points = np.array(json.load(f), dtype=np.float32)
        else:
            points = np.array(json.loads(args.points), dtype=np.float32)
        
        # Create simple segmentation
        output_path = os.path.splitext(args.frame)[0] + '_segmented.png'
        success = create_simple_segmentation(args.frame, points, output_path)
        
        if not success:
            sys.exit(1)
        return
    
    print("Error: Must provide either --input/--output for frame extraction or --frame/--points/--frame_dir for segmentation")
    sys.exit(1)

if __name__ == "__main__":
    main()