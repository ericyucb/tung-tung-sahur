import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import tempfile
import shutil

# Set matplotlib to use non-interactive backend for server environment
plt.switch_backend('Agg')

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import SAM2 after ensuring dependencies are available
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("SAM2 not available. Please install with: pip install git+https://github.com/facebookresearch/sam2.git")
    build_sam2_video_predictor = None

class SAM2Processor:
    def __init__(self, checkpoint_path=None, config_path=None):
        self.device = self._get_device()
        self.predictor = None
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        if build_sam2_video_predictor and checkpoint_path and config_path:
            self._initialize_predictor()
    
    def _get_device(self):
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Configure CUDA optimizations
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS device detected. SAM2 performance may vary on MPS.")
        else:
            device = torch.device("cpu")
        
        print(f"Using device: {device}")
        return device
    
    def _initialize_predictor(self):
        """Initialize the SAM2 predictor with the specified model."""
        try:
            self.predictor = build_sam2_video_predictor(
                self.config_path, 
                self.checkpoint_path, 
                device=self.device
            )
            print("✅ SAM2 predictor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize SAM2 predictor: {e}")
            self.predictor = None
    
    def extract_frames_from_video(self, video_path, output_dir):
        """Extract frames from video and save them as numbered JPG files."""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_filename = f"{frame_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {frame_count} frames to {output_dir}")
        return frame_count
    
    def process_video_with_points(self, video_path, points, output_dir):
        """Process video with SAM2 using provided points for segmentation."""
        if not self.predictor:
            raise ValueError("SAM2 predictor not initialized")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames
            frame_dir = os.path.join(temp_dir, "frames")
            self.extract_frames_from_video(video_path, frame_dir)
            
            # Initialize inference state
            inference_state = self.predictor.init_state(video_path=frame_dir)
            
            # Set up points and labels
            labels = np.array([1] * len(points), dtype=np.int32)
            points_np = np.array(points, dtype=np.float32)
            
            # Add points to the first frame
            ann_frame_idx = 0
            ann_obj_id = 1
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_np,
                labels=labels,
            )
            
            # Propagate masks through video
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # Create masked frames
            masked_frame_dir = os.path.join(temp_dir, "masked_frames")
            os.makedirs(masked_frame_dir, exist_ok=True)
            
            frame_names = sorted(
                [f for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")],
                key=lambda x: int(os.path.splitext(x)[0])
            )
            
            # Save masked frames
            for idx, fname in enumerate(tqdm(frame_names, desc="Processing frames")):
                img_path = os.path.join(frame_dir, fname)
                img = np.array(Image.open(img_path).convert("RGB"))
                masks = video_segments.get(idx, {})
                
                if masks:
                    mask = list(masks.values())[0].squeeze().astype(bool)
                    masked_img = img.copy()
                    masked_img[~mask] = 0
                    Image.fromarray(masked_img).save(os.path.join(masked_frame_dir, fname))
                else:
                    # If no mask for this frame, save original
                    Image.fromarray(img).save(os.path.join(masked_frame_dir, fname))
            
            # Create output video
            os.makedirs(output_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"{video_name}_masked.mp4")
            
            # Get video properties from first frame
            first_frame = cv2.imread(os.path.join(masked_frame_dir, frame_names[0]))
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
            
            # Write frames to video
            for fname in frame_names:
                fpath = os.path.join(masked_frame_dir, fname)
                if os.path.exists(fpath):
                    frame = cv2.imread(fpath)
                    out_video.write(frame)
            
            out_video.release()
            print(f"✅ Masked video saved at: {output_video_path}")
            
            return output_video_path
    
    def preview_mask(self, video_path, points, output_path=None):
        """Generate a preview of the mask for the first frame."""
        if not self.predictor:
            raise ValueError("SAM2 predictor not initialized")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract first frame
            frame_dir = os.path.join(temp_dir, "frames")
            self.extract_frames_from_video(video_path, frame_dir)
            
            # Initialize inference state
            inference_state = self.predictor.init_state(video_path=frame_dir)
            
            # Set up points and labels
            labels = np.array([1] * len(points), dtype=np.int32)
            points_np = np.array(points, dtype=np.float32)
            
            # Add points to the first frame
            ann_frame_idx = 0
            ann_obj_id = 1
            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points_np,
                labels=labels,
            )
            
            # Generate preview image
            mask_preview = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            img = Image.open(os.path.join(frame_dir, "00000.jpg"))
            img_np = np.array(img)
            
            # Create preview with mask overlay
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("SAM2 Segmentation Preview")
            ax.imshow(img_np)
            ax.scatter(points_np[:, 0], points_np[:, 1], c='lime', marker='*', s=300, edgecolors='white')
            
            # Overlay mask
            h, w = mask_preview.shape[:2]
            mask_rgb = np.zeros((h, w, 4))
            mask_rgb[:, :, 1] = 1  # green
            mask_rgb[:, :, 3] = 0.5 * mask_preview
            ax.imshow(mask_rgb)
            
            plt.axis("on")
            plt.grid(True)
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"✅ Preview saved at: {output_path}")
            else:
                plt.show()
            
            return inference_state, ann_obj_id

# Default model paths (you'll need to download these)
DEFAULT_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

def create_sam2_processor(checkpoint_path=None, config_path=None):
    """Factory function to create a SAM2 processor instance."""
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT
    if config_path is None:
        config_path = DEFAULT_CONFIG
    
    return SAM2Processor(checkpoint_path, config_path) 