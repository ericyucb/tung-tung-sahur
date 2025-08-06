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
        sam2_dir = os.path.expanduser("~/models/sam2")
        
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

def process_video_with_sam2(video_path, points, output_video_path):
    """Process entire video with SAM2 video predictor to create masked video with OpenPose landmarks."""
    try:
        # Get the SAM2 directory
        sam2_dir = os.path.expanduser("~/models/sam2")
        
        # Store current directory
        original_dir = os.getcwd()
        
        # Change directory to SAM2 root before loading the config
        os.chdir(sam2_dir)
        print(f"Changed working directory to: {os.getcwd()}")
        
        # SAM2 video predictor setup
        ckpt_path = 'checkpoints/sam2.1_hiera_tiny.pt'
        device = 'cuda'  # Use GPU by default
        
        print(f"Using checkpoint: {ckpt_path}")
        print(f"Using device: {device}")
        
        # Use SAM2 video predictor for full video processing
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)
        
        # Convert points to the format expected by SAM2
        points_np = np.array(points)
        labels = np.ones(len(points), dtype=np.int32)  # All positive points
        
        print(f"Processing video with points: {points_np}")
        print(f"Labels: {labels}")
        
        # Extract video frames using ffmpeg (inspired by allcode.py)
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = f"/tmp/{video_id}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        print(f"Extracting frames to: {frame_dir}")
        # Extract frames using ffmpeg
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-vf', 'fps=30',  # Extract at 30fps
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
        
        # Initialize state with frame directory (inspired by allcode.py)
        print("Initializing video state with frame directory...")
        state = predictor.init_state(video_path=frame_dir)
        
        # Add points to the first frame (frame 0)
        print("Adding points to first frame...")
        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            state, 
            frame_idx=0, 
            obj_id=1, 
            points=points_np, 
            labels=labels
        )
        
        print(f"Generated masks for frame {frame_idx}, object IDs: {obj_ids}")
        
        # Propagate masks through video (inspired by allcode.py)
        print("Propagating masks through video...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Create masked frames directory
        masked_frame_dir = f"/tmp/{video_id}_masked_frames"
        os.makedirs(masked_frame_dir, exist_ok=True)
        
        # Collect frame names
        frame_names = sorted(
            [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        print(f"Processing {len(frame_names)} frames...")
        
        # Save masked frames (inspired by allcode.py)
        for idx, fname in enumerate(frame_names):
            img_path = os.path.join(frame_dir, fname)
            img = np.array(Image.open(img_path).convert("RGB"))
            masks = video_segments.get(idx, {})
            
            if masks:
                mask = list(masks.values())[0].squeeze().astype(bool)
                if mask.shape == img.shape[:2]:
                    masked_img = img.copy()
                    masked_img[~mask] = 0
                    Image.fromarray(masked_img).save(os.path.join(masked_frame_dir, fname))
                else:
                    # If mask shape doesn't match, save original frame
                    Image.fromarray(img).save(os.path.join(masked_frame_dir, fname))
            else:
                # If no mask, save black frame
                black_img = np.zeros_like(img)
                Image.fromarray(black_img).save(os.path.join(masked_frame_dir, fname))
        
        # Combine masked frames into video using ffmpeg
        print("Combining masked frames into video...")
        if frame_names:
            # Use ffmpeg to create a more compatible video
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-framerate', str(fps),
                '-i', os.path.join(masked_frame_dir, '%05d.jpg'),
                '-c:v', 'libx264',  # Use H.264 codec for better browser compatibility
                '-preset', 'fast',
                '-crf', '23',  # Good quality
                '-pix_fmt', 'yuv420p',  # Standard pixel format
                output_video_path
            ]
            
            try:
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"✅ Final masked video saved at: {output_video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error creating video with ffmpeg: {e}")
                print(f"ffmpeg stderr: {e.stderr.decode()}")
                return False
        else:
            print("Error: No frames found")
            return False
        
        # Clean up temporary directories
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        shutil.rmtree(masked_frame_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"Error processing video with SAM2: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Ensure we change back to original directory
        os.chdir(original_dir)

def run_openpose_on_masked_frames(masked_frame_dir, output_json_dir, output_video_path):
    """Run OpenPose on masked frames to create video with landmarks."""
    try:
        print(f"Running OpenPose on masked frames in: {masked_frame_dir}")
        print(f"JSON output directory: {output_json_dir}")
        print(f"Video output path: {output_video_path}")
        
        # Create output directories
        os.makedirs(output_json_dir, exist_ok=True)
        
        # Import subprocess at the top of the function
        import subprocess
        
        # Add OpenPose Python module to path
        openpose_path = os.path.expanduser("~/models/openpose/build/python")
        sys.path.append(openpose_path)
        sys.path.append(os.path.join(openpose_path, "openpose"))
        
        # Import OpenPose Python module
        try:
            import pyopenpose as op
        except ImportError:
            # Try alternative import path
            sys.path.append(os.path.expanduser("~/models/openpose/build/python/openpose"))
            import pyopenpose as op
        
        # Configure OpenPose - only write JSON, not video
        params = dict()
        params["model_folder"] = os.path.expanduser("~/models/openpose/models/")
        params["number_people_max"] = 1
        params["net_resolution"] = "-1x368"
        params["write_json"] = output_json_dir
        params["display"] = 0
        
        # Initialize OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        
        # Process frames and collect results
        frame_files = sorted([f for f in os.listdir(masked_frame_dir) if f.lower().endswith(('.jpg', '.png'))])
        processed_frames = []
        
        for frame_file in frame_files:
            frame_path = os.path.join(masked_frame_dir, frame_file)
            datum = op.Datum()
            imageToProcess = cv2.imread(frame_path)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Get the processed image with landmarks
            if datum.cvOutputData is not None:
                processed_frames.append(datum.cvOutputData)
            else:
                # If no landmarks detected, use original frame
                processed_frames.append(imageToProcess)
        
        # Create video from processed frames using ffmpeg
        if processed_frames:
            # Save processed frames temporarily
            temp_frame_dir = f"/tmp/openpose_processed_frames_{os.getpid()}"
            os.makedirs(temp_frame_dir, exist_ok=True)
            
            for i, frame in enumerate(processed_frames):
                frame_path = os.path.join(temp_frame_dir, f"{i:05d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # Use ffmpeg to create video
            subprocess.run([
                'ffmpeg', '-y',
                '-framerate', '30',
                '-i', os.path.join(temp_frame_dir, '%05d.jpg'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                output_video_path
            ], check=True)
            
            # Clean up temp frames
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
            
            print("✅ OpenPose processing complete!")
            return True
        else:
            print("❌ No frames were processed")
            return False
            
    except ImportError as e:
        print(f"OpenPose import error: {e}")
        print("Trying to install OpenPose Python module...")
        try:
            # Try to build the Python module if it's missing
            import subprocess
            subprocess.run([
                'cd', os.path.expanduser("~/models/openpose/build/python"),
                '&&', 'make', 'pyopenpose'
            ], check=True, shell=True)
            # Try import again
            import pyopenpose as op
        except Exception as build_error:
            print(f"Failed to build OpenPose Python module: {build_error}")
            return False
    except Exception as e:
        print(f"Error running OpenPose: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='SAM2 Video Processing')
    parser.add_argument('--input', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output frame file path')
    parser.add_argument('--frame', type=str, help='Input frame file path for segmentation')
    parser.add_argument('--points', type=str, help='Points for segmentation (JSON string)')
    parser.add_argument('--frame_dir', type=str, help='Frame directory for output')
    parser.add_argument('--video', type=str, help='Input video file path for full video processing')
    parser.add_argument('--video_output', type=str, help='Output video file path for masked video')
    parser.add_argument('--openpose', type=str, help='Run OpenPose on masked video file')
    
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
    
    # Full video processing mode
    elif args.video and args.points and args.video_output:
        points = json.loads(args.points)
        success = process_video_with_sam2(args.video, points, args.video_output)
        
        if not success:
            sys.exit(1)
    
    # OpenPose mode
    elif args.openpose:
        masked_video_path = args.openpose
        if not os.path.exists(masked_video_path):
            print(f"Error: Masked video not found: {masked_video_path}")
            sys.exit(1)
        
        # Create output paths
        output_video_path = masked_video_path.replace('.mp4', '_with_landmarks.mp4')
        
        # Extract frames from masked video
        video_id = os.path.splitext(os.path.basename(masked_video_path))[0]
        frame_dir = f"/tmp/{video_id}_masked_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        # Extract frames using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', masked_video_path, 
            '-vf', 'fps=30',
            '-frame_pts', '1',
            os.path.join(frame_dir, '%05d.jpg')
        ], check=True)
        
        # Run OpenPose on masked frames
        openpose_json_dir = f"/tmp/{video_id}_openpose_json"
        
        openpose_success = run_openpose_on_masked_frames(
            frame_dir, 
            openpose_json_dir, 
            output_video_path
        )
        
        if openpose_success:
            print(f"✅ OpenPose processing completed: {output_video_path}")
        else:
            print("❌ OpenPose processing failed")
            sys.exit(1)
    
    else:
        print("Usage:")
        print("  For frame extraction: --input <video> --output <frame>")
        print("  For segmentation: --frame <frame> --points <json> --frame_dir <dir>")
        print("  For full video processing: --video <video> --points <json> --video_output <output>")
        print("  For OpenPose processing: --openpose <masked_video_path>")
        sys.exit(1)

if __name__ == "__main__":
    main()