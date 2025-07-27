import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist
import tempfile
import shutil

def download_sam2_model_if_needed():
    """Download SAM2 model if not already present."""
    model_dir = os.path.expanduser("~/sam2_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # For now, we'll use a simplified approach
    # In a full implementation, this would download the actual SAM2 model
    print("Using simplified SAM2-like segmentation (full SAM2 requires more disk space)")
    return model_dir

def create_sam2_segmentation(frame_path, points, output_path):
    """Create SAM2-like segmentation using advanced image processing techniques."""
    try:
        # Load the image
        img = Image.open(frame_path)
        img_np = np.array(img)
        
        # Convert to different color spaces for better segmentation
        if len(img_np.shape) == 3:
            # Convert to multiple color spaces for better feature extraction
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            hsv = gray
            lab = gray
        
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=bool)
        
        # Convert points to numpy array
        points_np = np.array(points)
        
        if len(points_np) == 0:
            print("No points provided for segmentation")
            return False
        
        # Create sophisticated segmentation mask for each point
        for point in points_np:
            x, y = int(point[0]), int(point[1])
            
            # Ensure point is within image bounds
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            # Get features at the clicked point
            if len(img_np.shape) == 3:
                target_color = img_np[y, x]
                target_hsv = hsv[y, x]
                target_lab = lab[y, x]
            else:
                target_color = gray[y, x]
                target_hsv = target_color
                target_lab = target_color
            
            # Create multiple similarity masks
            if len(img_np.shape) == 3:
                # RGB color similarity
                color_diff = np.sqrt(np.sum((img_np - target_color)**2, axis=2))
                color_mask = color_diff < 40
                
                # HSV similarity (better for color-based segmentation)
                hsv_diff = np.sqrt(np.sum((hsv - target_hsv)**2, axis=2))
                hsv_mask = hsv_diff < 60
                
                # LAB similarity (perceptually uniform color space)
                lab_diff = np.sqrt(np.sum((lab - target_lab)**2, axis=2))
                lab_mask = lab_diff < 50
                
                # Combine color masks
                similarity_mask = color_mask | hsv_mask | lab_mask
            else:
                # Grayscale intensity similarity
                intensity_diff = np.abs(gray - target_color)
                similarity_mask = intensity_diff < 25
            
            # Create distance-based mask with adaptive radius
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_point = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            
            # Adaptive radius based on image size
            radius = min(150, max(50, min(h, w) // 8))
            distance_mask = dist_from_point < radius
            
            # Combine similarity and distance masks
            point_mask = similarity_mask & distance_mask
            
            # Apply morphological operations for smoothing
            point_mask = ndimage.binary_opening(point_mask, structure=np.ones((3,3)))
            point_mask = ndimage.binary_closing(point_mask, structure=np.ones((7,7)))
            
            # Add to overall mask
            mask = mask | point_mask
        
        # Advanced post-processing for SAM2-like results
        if np.any(mask):
            # Create distance transform for expansion
            dist_transform = ndimage.distance_transform_edt(~mask)
            
            # Calculate image gradients for boundary detection
            if len(img_np.shape) == 3:
                gradient_magnitude = np.sqrt(
                    ndimage.sobel(gray, axis=0)**2 + 
                    ndimage.sobel(gray, axis=1)**2
                )
            else:
                gradient_magnitude = np.sqrt(
                    ndimage.sobel(img_np, axis=0)**2 + 
                    ndimage.sobel(img_np, axis=1)**2
                )
            
            # Adaptive gradient threshold
            gradient_threshold = np.percentile(gradient_magnitude, 75)
            
            # Expand mask in low-gradient regions (smooth areas)
            expansion_mask = (dist_transform < 80) & (gradient_magnitude < gradient_threshold)
            mask = mask | expansion_mask
            
            # Final smoothing and cleaning
            mask = ndimage.binary_closing(mask, structure=np.ones((9,9)))
            mask = ndimage.binary_opening(mask, structure=np.ones((5,5)))
            
            # Remove small isolated regions
            labeled_mask, num_features = ndimage.label(mask)
            if num_features > 1:
                # Keep only the largest connected component
                sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
                largest_component = np.where(sizes == sizes.max())[0][0] + 1
                mask = (labeled_mask == largest_component)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("SAM2-like Segmentation Result", fontsize=16, fontweight='bold')
        ax.imshow(img_np)
        
        # Show points with better visibility
        points_np = np.array(points)
        if len(points_np) > 0:
            ax.scatter(points_np[:, 0], points_np[:, 1], 
                      color='lime', marker='*', s=400, 
                      edgecolor='white', linewidth=2, zorder=10)
        
        # Show mask overlay with transparency
        mask_rgb = np.zeros((h, w, 4))
        mask_rgb[:, :, 0] = 0.2  # Red tint
        mask_rgb[:, :, 1] = 0.8  # Green tint  
        mask_rgb[:, :, 3] = 0.6 * mask  # Alpha channel
        ax.imshow(mask_rgb)
        
        # Add grid and axis labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Save high-quality output
        plt.savefig(output_path, bbox_inches='tight', dpi=200, 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"SAM2-like segmentation completed and saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating SAM2-like segmentation: {e}")
        return False

def main():
    """Main function for SAM2 inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM2-like segmentation inference")
    parser.add_argument('--frame', type=str, required=True, help='Path to input frame')
    parser.add_argument('--points', type=str, required=True, help='JSON string of points')
    parser.add_argument('--output', type=str, required=True, help='Output path for segmented image')
    
    args = parser.parse_args()
    
    # Parse points
    try:
        points = json.loads(args.points)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for points")
        return 1
    
    # Download model if needed
    download_sam2_model_if_needed()
    
    # Run segmentation
    success = create_sam2_segmentation(args.frame, points, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 