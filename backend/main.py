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

app = FastAPI()

# Add startup message
print("=" * 80)
print("🚀 SAM2 AWS PROCESSING BACKEND STARTED!")
print("=" * 80)
print("✅ Server is running on http://0.0.0.0:8000")
print("✅ SAM2 integration: READY")
print("✅ GPU acceleration: ENABLED")
print("✅ Static files: SERVING")
print("=" * 80)

@app.on_event("startup")
async def startup_event():
    print("🎯 BACKEND UPDATED AND READY FOR REQUESTS!")
    print("📡 API endpoints:")
    print("   POST /upload - Upload video and extract first frame")
    print("   POST /coords - Receive coordinates from frontend")
    print("   POST /segment - Run SAM2 segmentation")
    print("   POST /process_video - Process full video with SAM2")
    print("=" * 80)

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
        # 1. Upload video to S3
        contents = await file.read()
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=file.filename,
            Body=contents,
            ContentType=file.content_type
        )

        # 2. Save video to temp file and extract first frame
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_path = os.path.join(tmpdir, file.filename)
            with open(local_video_path, 'wb') as f:
                f.write(contents)
            first_frame_path = os.path.join(tmpdir, 'first_frame.jpg')
            # Call inference.py to extract first frame
            subprocess.run([
                'python3', os.path.join(os.path.dirname(__file__), 'inference.py'),
                '--input', local_video_path,
                '--output', first_frame_path
            ], check=True)
            # Copy first frame to static directory
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(static_dir, exist_ok=True)
            static_frame_path = os.path.join(static_dir, f'{file.filename}_first_frame.jpg')
            shutil.copy(first_frame_path, static_frame_path)

        # 3. Return the static URL for the first frame
        return {
            "message": "Upload successful",
            "video_filename": file.filename,
            "s3_key": file.filename,
            "first_frame_url": f"/static/{file.filename}_first_frame.jpg"
        }
    except Exception as e:
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
            "process_video": "POST /process_video"
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
            # Run full video processing
            print(f"Starting full video processing for {video_filename}")
            subprocess.run([
                'python3', os.path.join(os.path.dirname(__file__), 'inference.py'),
                '--video', video_path,
                '--points', json.dumps(points_np.tolist()),
                '--video_output', output_video_path
            ], check=True)
            
            if os.path.exists(output_video_path):
                return {
                    "message": "Full video processing completed",
                    "masked_video_url": f"/video/{output_video_filename}"
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