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
import argparse
import subprocess
import boto3
from dotenv import load_dotenv
import torch

load_dotenv()

# S3 configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def upload_file_to_s3(local_path, s3_key):
    """Upload a file to S3."""
    try:
        with open(local_path, 'rb') as f:
            s3_client.put_object(
                Bucket=AWS_S3_BUCKET_NAME,
                Key=s3_key,
                Body=f.read()
            )
        print(f"✅ Uploaded to S3: {s3_key}")
        return True
    except Exception as e:
        print(f"❌ S3 upload error for {s3_key}: {e}")
        return False

def save_landmark_json(landmarks, frame_number, output_dir):
    """Save landmark coordinates to a JSON file."""
    if landmarks is None or len(landmarks) == 0:
        return None
    
    # OpenPose keypoint names (BODY_25 model)
    keypoint_names = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "mid_hip", "right_hip",
        "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe",
        "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"
    ]
    
    # Process the first person's landmarks (assuming single person)
    person_landmarks = landmarks[0]  # First person
    
    landmark_data = {
        "frame_number": frame_number,
        "timestamp": frame_number / 30.0,  # Assuming 30 fps
        "landmarks": []
    }
    
    for i, (x, y, confidence) in enumerate(person_landmarks):
        if confidence > 0.1:  # Only include landmarks with sufficient confidence
            landmark_data["landmarks"].append({
                "keypoint_id": i,
                "keypoint_name": keypoint_names[i] if i < len(keypoint_names) else f"keypoint_{i}",
                "x": float(x),
                "y": float(y),
                "confidence": float(confidence)
            })
    
    # Save to JSON file
    json_filename = f"frame_{frame_number:05d}_landmarks.json"
    json_path = os.path.join(output_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(landmark_data, f, indent=2)
    
    return json_path

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

def run_openpose_on_masked_video(masked_video_path, s3_folder=None):
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
        
        # Create JSON output directory
        json_output_dir = f"/tmp/{video_id}_landmark_jsons"
        os.makedirs(json_output_dir, exist_ok=True)
        
        print(f"Processing {len(frame_files)} frames with OpenPose...")
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frame_dir, frame_file)
            datum = op.Datum()
            image = cv2.imread(frame_path)
            datum.cvInputData = image
            
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Save landmark JSON for this frame
            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                json_path = save_landmark_json(datum.poseKeypoints, frame_idx, json_output_dir)
                if json_path:
                    print(f"✅ Saved landmarks for frame {frame_idx}: {json_path}")
            
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
            
            # Upload to S3 if s3_folder is provided
            if s3_folder:
                # Upload landmark video
                landmark_video_s3_key = f"{s3_folder}/landmark_video/{os.path.basename(output_video_path)}"
                upload_file_to_s3(output_video_path, landmark_video_s3_key)
                
                # Upload JSON files
                json_files = [f for f in os.listdir(json_output_dir) if f.endswith('.json')]
                for json_file in json_files:
                    json_path = os.path.join(json_output_dir, json_file)
                    json_s3_key = f"{s3_folder}/landmark_jsons/{json_file}"
                    upload_file_to_s3(json_path, json_s3_key)
                
                print(f"✅ Uploaded {len(json_files)} JSON files to S3")
            
            # Clean up
            shutil.rmtree(frame_dir, ignore_errors=True)
            shutil.rmtree(temp_frame_dir, ignore_errors=True)
            shutil.rmtree(json_output_dir, ignore_errors=True)
            
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
    parser.add_argument('--s3_folder', type=str, help='S3 folder for storing outputs')
    
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
        success = run_openpose_on_masked_video(args.openpose, args.s3_folder)
        if not success:
            sys.exit(1)
    
    else:
        print("❌ Invalid arguments. Use --help for usage information.")
        sys.exit(1)
    
    print("✅ Processing completed successfully!")

if __name__ == "__main__":
    main()