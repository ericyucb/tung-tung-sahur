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
import subprocess

# Add SAM2 to path for imports
sys.path.append(os.path.expanduser("~/models/sam2"))

# SAM2 imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
        
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            cap.release()
            return False
        
        success = cv2.imwrite(output_path, frame)
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
    """Create segmentation using SAM2 model on a single frame."""
    try:
        sam2_dir = os.path.expanduser("~/models/sam2")
        original_dir = os.getcwd()
        
        os.chdir(sam2_dir)
        print(f"Changed working directory to: {os.getcwd()}")
        
        ckpt_path = 'checkpoints/sam2.1_hiera_tiny.pt'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using checkpoint: {ckpt_path}")
        print(f"Using device: {device}")
        
        # Load SAM2 model for single image
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        model = build_sam2(model_cfg, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(model)
        
        # Convert points to SAM2 format
        points_np = np.array(points)
        labels = np.ones(len(points), dtype=np.int32)  # All positive points
        
        print(f"Points: {points_np}")
        print(f"Labels: {labels}")
        
        # Load and process image
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Could not load image: {frame_path}")
        
        predictor.set_image(img)
        
        # Run segmentation
        masks, scores, logits = predictor.predict(
            point_coords=points_np,
            point_labels=labels
        )
        
        # Create visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("SAM2 Segmentation Preview")
        ax.imshow(img_rgb)
        
        # Show points
        if len(points_np) > 0:
            ax.scatter(points_np[:, 0], points_np[:, 1], color='lime', marker='*', s=300, edgecolors='white', linewidths=2)
        
        # Show mask overlay
        if masks is not None and len(masks) > 0:
            mask = masks[0]  # Use the first mask
            mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
            mask_overlay[:, :, 1] = 1  # Green color
            mask_overlay[:, :, 3] = 0.4 * mask  # Semi-transparent
            ax.imshow(mask_overlay)
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        
        print(f"SAM2 segmentation preview saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating SAM2 segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

def process_video_with_sam2(video_path, points, output_video_path):
    """Apply SAM2 masking to entire video."""
    try:
        sam2_dir = os.path.expanduser("~/models/sam2")
        original_dir = os.getcwd()
        
        os.chdir(sam2_dir)
        print(f"Changed working directory to: {os.getcwd()}")
        
        ckpt_path = 'checkpoints/sam2.1_hiera_tiny.pt'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using checkpoint: {ckpt_path}")
        print(f"Using device: {device}")
        
        # Load SAM2 video predictor
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)
        
        # Convert points to SAM2 format
        points_np = np.array(points)
        labels = np.ones(len(points), dtype=np.int32)
        
        print(f"Processing video with points: {points_np}")
        
        # Extract frames
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = f"/tmp/{video_id}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        print(f"Extracting frames to: {frame_dir}")
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-vf', 'fps=30',
            '-frame_pts', '1',
            os.path.join(frame_dir, '%05d.jpg')
        ], check=True)
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Video properties: {fps} fps, {width}x{height}")
        
        # Initialize SAM2 state
        print("Initializing SAM2 video state...")
        state = predictor.init_state(video_path=frame_dir)
        
        # Add points to first frame
        print("Adding points to first frame...")
        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            state, 
            frame_idx=0, 
            obj_id=1, 
            points=points_np, 
            labels=labels
        )
        
        print(f"Generated masks for frame {frame_idx}, object IDs: {obj_ids}")
        
        # Propagate masks through video
        print("Propagating masks through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Create masked frames
        masked_frame_dir = f"/tmp/{video_id}_masked_frames"
        os.makedirs(masked_frame_dir, exist_ok=True)
        
        frame_names = sorted(
            [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        print(f"Processing {len(frame_names)} frames...")
        
        # Apply masks to frames
        for idx, fname in enumerate(frame_names):
            img_path = os.path.join(frame_dir, fname)
            img = np.array(Image.open(img_path).convert("RGB"))
            masks = video_segments.get(idx, {})
            
            if masks:
                mask = list(masks.values())[0].squeeze().astype(bool)
                if mask.shape == img.shape[:2]:
                    masked_img = img.copy()
                    masked_img[~mask] = 0  # Black out non-masked areas
                    Image.fromarray(masked_img).save(os.path.join(masked_frame_dir, fname))
                else:
                    Image.fromarray(img).save(os.path.join(masked_frame_dir, fname))
            else:
                # No mask - create black frame
                black_img = np.zeros_like(img)
                Image.fromarray(black_img).save(os.path.join(masked_frame_dir, fname))
        
        # Create video from masked frames
        print("Creating masked video...")
        if frame_names:
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(masked_frame_dir, '%05d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_video_path
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"✅ Masked video saved at: {output_video_path}")
        else:
            print("Error: No frames found")
            return False
        
        # Clean up temporary directories
        shutil.rmtree(frame_dir, ignore_errors=True)
        shutil.rmtree(masked_frame_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"Error processing video with SAM2: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

def run_openpose_on_masked_video(masked_video_path):
    """Run OpenPose on a masked video to add pose landmarks."""
    try:
        print(f"Running OpenPose on masked video: {masked_video_path}")
        
        # Create output path
        output_video_path = masked_video_path.replace('.mp4', '_with_landmarks.mp4')
        
        # Extract frames from masked video
        video_id = os.path.splitext(os.path.basename(masked_video_path))[0]
        frame_dir = f"/tmp/{video_id}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        print(f"Extracting frames from masked video to: {frame_dir}")
        subprocess.run([
            'ffmpeg', '-i', masked_video_path, 
            '-vf', 'fps=30',
            '-frame_pts', '1',
            os.path.join(frame_dir, '%05d.jpg')
        ], check=True)
        
        # Get video properties
        cap = cv2.VideoCapture(masked_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Setup OpenPose
        openpose_path = os.path.expanduser("~/models/openpose/build/python")
        sys.path.append(openpose_path)
        sys.path.append(os.path.join(openpose_path, "openpose"))
        
        try:
            import pyopenpose as op
        except ImportError:
            sys.path.append(os.path.expanduser("~/models/openpose/build/python/openpose"))
            import pyopenpose as op
        
        # Configure OpenPose
        params = dict()
        params["model_folder"] = os.path.expanduser("~/models/openpose/models/")
        params["number_people_max"] = 1
        params["net_resolution"] = "-1x368"
        params["display"] = 0
        
        # Initialize OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        # Process frames
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith('.jpg')])
        processed_frames = []
        
        print(f"Processing {len(frame_files)} frames with OpenPose...")
        
        for frame_file in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)
            datum = op.Datum()
            image = cv2.imread(frame_path)
            datum.cvInputData = image
            
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Get processed frame with landmarks
            if datum.cvOutputData is not None:
                processed_frames.append(datum.cvOutputData)
            else:
                # If no landmarks detected, use original frame
                processed_frames.append(image)
        
        # Create video from processed frames
        if processed_frames:
            temp_frame_dir = f"/tmp/{video_id}_openpose_frames"
            os.makedirs(temp_frame_dir, exist_ok=True)
            
            # Save processed frames
            for i, frame in enumerate(processed_frames):
                frame_path = os.path.join(temp_frame_dir, f"{i:05d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # Create final video with landmarks
            subprocess.run([
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_frame_dir, '%05d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_video_path
            ], check=True)
            
            print(f"✅ OpenPose processing complete: {output_video_path}")
            
            # Clean up
            shutil.rmtree(frame_dir, ignore_errors=True)
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
            
            return True
        else:
            print("❌ No frames were processed")
            return False
            
    except Exception as e:
        print(f"Error running OpenPose: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point with clear workflow steps."""
    parser = argparse.ArgumentParser(description='SAM2 + OpenPose Video Processing Pipeline')
    
    # Step 1: Extract first frame
    parser.add_argument('--input', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output first frame path')
    
    # Step 2: SAM2 segmentation on first frame  
    parser.add_argument('--frame', type=str, help='Input frame file path for segmentation')
    parser.add_argument('--points', type=str, help='Points for segmentation (JSON string)')
    parser.add_argument('--frame_dir', type=str, help='Frame directory for output')
    
    # Step 3: Apply SAM2 to full video
    parser.add_argument('--video', type=str, help='Input video file path for full video processing')
    parser.add_argument('--video_output', type=str, help='Output masked video file path')
    
    # Step 4: Run OpenPose on masked video
    parser.add_argument('--openpose', type=str, help='Masked video file path for OpenPose processing')
    
    args = parser.parse_args()
    
    # Step 1: Extract first frame from video
    if args.input and args.output:
        print("🎬 STEP 1: Extracting first frame from video")
        success = extract_first_frame(args.input, args.output)
        if not success:
            sys.exit(1)
    
    # Step 2: Create SAM2 segmentation preview on first frame
    elif args.frame and args.points and args.frame_dir:
        print("🎯 STEP 2: Creating SAM2 segmentation preview")
        points = json.loads(args.points)
        output_path = os.path.splitext(args.frame)[0] + '_segmented.png'
        success = create_sam2_segmentation(args.frame, points, output_path)
        if not success:
            sys.exit(1)
    
    # Step 3: Apply SAM2 masking to full video
    elif args.video and args.points and args.video_output:
        print("🎥 STEP 3: Applying SAM2 masking to full video")
        points = json.loads(args.points)
        success = process_video_with_sam2(args.video, points, args.video_output)
        if not success:
            sys.exit(1)
    
    # Step 4: Add OpenPose landmarks to masked video
    elif args.openpose:
        print("🕺 STEP 4: Adding OpenPose landmarks to masked video")
        success = run_openpose_on_masked_video(args.openpose)
        if not success:
            sys.exit(1)
    
    else:
        print("📋 SAM2 + OpenPose Processing Pipeline")
        print()
        print("Usage:")
        print("  Step 1 - Extract first frame:")
        print("    --input <video> --output <frame>")
        print()
        print("  Step 2 - Preview SAM2 segmentation:")
        print("    --frame <frame> --points <json> --frame_dir <dir>")
        print()  
        print("  Step 3 - Apply SAM2 to full video:")
        print("    --video <video> --points <json> --video_output <output>")
        print()
        print("  Step 4 - Add OpenPose landmarks:")
        print("    --openpose <masked_video_path>")
        print()
        print("Complete workflow:")
        print("  1. Upload video → Extract first frame")
        print("  2. User clicks points → Show SAM2 preview") 
        print("  3. Apply SAM2 to full video → Create masked video")
        print("  4. Run OpenPose on masked video → Add pose landmarks")
        sys.exit(1)

if __name__ == "__main__":
    main()