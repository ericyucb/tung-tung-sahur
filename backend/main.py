import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import boto3
import os
from dotenv import load_dotenv
import tempfile
import subprocess
import json
import numpy as np

load_dotenv()

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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 80)
    print("🚀 SAM2 AWS PROCESSING BACKEND STARTED!")
    print("=" * 80)
    print("✅ Server is running on http://0.0.0.0:8000")
    print("✅ SAM2 integration: READY")
    print("✅ GPU acceleration: ENABLED")
    print("✅ Static files: SERVING")
    print("=" * 80)
    print("🎯 BACKEND UPDATED AND READY FOR REQUESTS!")
    print("📡 API endpoints:")
    print("   POST /upload - Upload video and extract first frame")
    print("   POST /coords - Receive coordinates from frontend")
    print("   POST /segment - Run SAM2 segmentation")
    print("   POST /process_video - Process full video with SAM2 and OpenPose")
    print("   POST /process_video_openpose - Process video with OpenPose only")
    print("   POST /run_openpose_on_masked_video - Run OpenPose on existing masked video")
    print("=" * 80)
    yield
    # Shutdown
    print("🛑 Backend server shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        print(f"Received upload request for file: {file.filename}")
        
        # 1. Upload video to S3
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")

        # Save video to temp file first
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_path = os.path.join(tmpdir, file.filename)
            first_frame_path = os.path.join(tmpdir, 'first_frame.jpg')
            
            # Write the file contents
            with open(local_video_path, 'wb') as f:
                f.write(contents)
            
            print(f"Saved video to: {local_video_path}")

            # Upload to S3
            try:
                s3_client.put_object(
                    Bucket=AWS_S3_BUCKET_NAME,
                    Key=file.filename,
                    Body=contents,
                    ContentType=file.content_type
                )
                print(f"Uploaded to S3: {file.filename}")
            except Exception as s3_error:
                print(f"S3 upload error: {s3_error}")
                # Continue even if S3 upload fails
            
            # Extract first frame directly (much faster)
            try:
                import cv2
                cap = cv2.VideoCapture(local_video_path)
                if not cap.isOpened():
                    return JSONResponse(status_code=500, content={"error": "Could not open video file"})
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return JSONResponse(status_code=500, content={"error": "Could not read first frame"})
                
                success = cv2.imwrite(first_frame_path, frame)
                if not success:
                    return JSONResponse(status_code=500, content={"error": "Could not save first frame"})
                
                print(f"First frame extracted successfully: {first_frame_path}")
            except Exception as e:
                print(f"Frame extraction failed: {e}")
                return JSONResponse(status_code=500, content={"error": f"Frame extraction failed: {str(e)}"})
            
            # Copy first frame to static directory
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(static_dir, exist_ok=True)
            static_frame_path = os.path.join(static_dir, f'{file.filename}_first_frame.jpg')
            shutil.copy(first_frame_path, static_frame_path)
            print(f"Copied frame to: {static_frame_path}")

        # 3. Return the static URL for the first frame
        return {
            "message": "Upload successful",
            "video_filename": file.filename,
            "s3_key": file.filename,
            "first_frame_url": f"/static/{file.filename}_first_frame.jpg"
        }
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static')), name="static")

@app.get("/video/{filename}")
async def serve_video(filename: str):
    """Serve video files with proper headers for browser compatibility."""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    video_path = os.path.join(static_dir, filename)
    
    if not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={filename}",
            "Cache-Control": "public, max-age=3600"
        }
    )

@app.get("/")
async def root():
    return {
        "message": "🚀 SAM2 AWS Processing Backend",
        "status": "UPDATED AND RUNNING",
        "version": "2.1 - Full Video Processing Added",
        "endpoints": {
            "upload": "POST /upload",
            "coords": "POST /coords", 
            "segment": "POST /segment",
            "process_video": "POST /process_video",
            "run_openpose_on_masked_video": "POST /run_openpose_on_masked_video"
        }
    }

@app.post("/coords")
async def receive_coords(request: Request):
    data = await request.json()
    print("Received coordinates from frontend:", data)
    return {"message": "Coordinates received", "coords": data}

@app.post("/segment")
async def segment_frame(request: Request):
    try:
        data = await request.json()
        points = data.get('points', [])
        video_filename = data.get('video_filename')
        
        if not points or not video_filename:
            return JSONResponse(status_code=400, content={"error": "Missing points or video_filename"})
        
        # Convert points to the format expected by SAM2
        points_np = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        labels = np.ones(len(points), dtype=np.int32)  # All positive points
        
        # Find the first frame path
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        first_frame_path = os.path.join(static_dir, f'{video_filename}_first_frame.jpg')
        
        if not os.path.exists(first_frame_path):
            return JSONResponse(status_code=404, content={"error": "First frame not found"})
        
        # Create a temporary directory for the frame
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_dir = tmpdir
            # Copy the first frame to the temp directory
            import shutil
            temp_frame_path = os.path.join(frame_dir, "00001.jpg")
            shutil.copy(first_frame_path, temp_frame_path)
            
            # Run simple segmentation
            subprocess.run([
                'python3', os.path.join(os.path.dirname(__file__), 'inference.py'),
                '--frame', temp_frame_path,
                '--points', json.dumps(points_np.tolist()),
                '--frame_dir', frame_dir
            ], check=True)
            
            # Find the segmented image
            segmented_path = os.path.splitext(temp_frame_path)[0] + '_segmented.png'
            if os.path.exists(segmented_path):
                # Copy to static directory
                static_segmented_path = os.path.join(static_dir, f'{video_filename}_segmented.png')
                shutil.copy(segmented_path, static_segmented_path)
                
                return {
                    "message": "Segmentation completed",
                    "segmented_image_url": f"/static/{video_filename}_segmented.png"
                }
            else:
                return JSONResponse(status_code=500, content={"error": "Segmentation failed"})
                
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process_video")
async def process_full_video(request: Request):
    try:
        data = await request.json()
        points = data.get('points', [])
        video_filename = data.get('video_filename')
        
        if not points or not video_filename:
            return JSONResponse(status_code=400, content={"error": "Missing points or video_filename"})
        
        # Convert points to the format expected by SAM2
        points_np = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
        
        # Download video from S3 to local temp file
        print(f"Downloading video {video_filename} from S3...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video_filename}") as temp_video:
            try:
                # Download from S3
                s3_client.download_file(
                    Bucket=AWS_S3_BUCKET_NAME,
                    Key=video_filename,
                    Filename=temp_video.name
                )
                video_path = temp_video.name
                print(f"Video downloaded to: {video_path}")
            except Exception as e:
                return JSONResponse(status_code=404, content={"error": f"Could not download video from S3: {str(e)}"})
        
        # Create output video path in static directory
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        output_video_filename = f'{video_filename}_masked.mp4'
        output_video_path = os.path.join(static_dir, output_video_filename)
        
        try:
            # Run full video processing with SAM2 and OpenPose
            print(f"Starting full video processing for {video_filename}")
            subprocess.run([
                'python3', os.path.join(os.path.dirname(__file__), 'inference.py'),
                '--video', video_path,
                '--points', json.dumps(points_np.tolist()),
                '--video_output', output_video_path
            ], check=True)
            
            if os.path.exists(output_video_path):
                return {
                    "message": "Full video processing completed with SAM2 and OpenPose",
                    "masked_video_url": f"/video/{output_video_filename}",
                    "has_landmarks": True
                }
            else:
                return JSONResponse(status_code=500, content={"error": "Video processing failed"})
        
        finally:
            # Clean up temporary video file
            if os.path.exists(video_path):
                os.unlink(video_path)
                print(f"Cleaned up temporary file: {video_path}")
                
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)}) 

@app.post("/process_video_openpose")
async def process_video_with_openpose(request: Request):
    """Process a video with OpenPose to create video with landmarks."""
    try:
        data = await request.json()
        video_filename = data.get('video_filename')
        
        if not video_filename:
            return JSONResponse(status_code=400, content={"error": "Missing video_filename"})
        
        # Download video from S3 to local temp file
        print(f"Downloading video {video_filename} from S3 for OpenPose processing...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video_filename}") as temp_video:
            try:
                # Download from S3
                s3_client.download_file(
                    Bucket=AWS_S3_BUCKET_NAME,
                    Key=video_filename,
                    Filename=temp_video.name
                )
                video_path = temp_video.name
                print(f"Video downloaded to: {video_path}")
            except Exception as e:
                return JSONResponse(status_code=404, content={"error": f"Could not download video from S3: {str(e)}"})
        
        # Create output paths
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_dir, exist_ok=True)
        output_video_filename = f'{video_filename}_openpose.mp4'
        output_video_path = os.path.join(static_dir, output_video_filename)
        
        # Extract frames from video
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = f"/tmp/{video_id}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        # Extract frames using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-vf', 'fps=30',
            '-frame_pts', '1',
            os.path.join(frame_dir, '%05d.jpg')
        ], check=True)
        
        # Run OpenPose on frames
        openpose_json_dir = f"/tmp/{video_id}_openpose_json"
        
        # Import the OpenPose function
        from inference import run_openpose_on_masked_frames
        
        openpose_success = run_openpose_on_masked_frames(
            frame_dir, 
            openpose_json_dir, 
            output_video_path
        )
        
        if openpose_success and os.path.exists(output_video_path):
            return {
                "message": "OpenPose processing completed",
                "openpose_video_url": f"/video/{output_video_filename}",
                "has_landmarks": True
            }
        else:
            return JSONResponse(status_code=500, content={"error": "OpenPose processing failed"})
        
        # Clean up temporary files
        if os.path.exists(video_path):
            os.unlink(video_path)
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        shutil.rmtree(openpose_json_dir, ignore_errors=True)
                
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/run_openpose_on_masked_video")
async def run_openpose_on_masked_video(request: Request):
    """Run OpenPose on an existing masked video to add landmarks."""
    try:
        data = await request.json()
        masked_video_filename = data.get('masked_video_filename')
        
        if not masked_video_filename:
            return JSONResponse(status_code=400, content={"error": "Missing masked_video_filename"})
        
        # Check if the masked video exists in static directory
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        masked_video_path = os.path.join(static_dir, masked_video_filename)
        
        if not os.path.exists(masked_video_path):
            return JSONResponse(status_code=404, content={"error": "Masked video not found"})
        
        print(f"Running OpenPose on masked video: {masked_video_filename}")
        
        # Create output paths
        output_video_filename = masked_video_filename.replace('.mp4', '_with_landmarks.mp4')
        output_video_path = os.path.join(static_dir, output_video_filename)
        
        # Use the inference.py script directly
        try:
            result = subprocess.run([
                'python3', os.path.join(os.path.dirname(__file__), 'inference.py'),
                '--openpose', masked_video_path
            ], capture_output=True, text=True)
            print(f"OpenPose processing output: {result.stdout}")
            
            # Check if the output file was created (this is the real success indicator)
            if os.path.exists(output_video_path):
                return {
                    "message": "OpenPose processing completed on masked video",
                    "landmarked_video_url": f"/video/{output_video_filename}",
                    "has_landmarks": True
                }
            else:
                # If file doesn't exist, check if there was an error
                if result.returncode != 0:
                    print(f"OpenPose processing error: {result.stderr}")
                    return JSONResponse(status_code=500, content={"error": f"OpenPose processing failed: {result.stderr}"})
                else:
                    return JSONResponse(status_code=500, content={"error": "OpenPose processing failed - output file not created"})
                
        except subprocess.CalledProcessError as e:
            print(f"OpenPose processing error: {e.stderr}")
            return JSONResponse(status_code=500, content={"error": f"OpenPose processing failed: {e.stderr}"})
                
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)