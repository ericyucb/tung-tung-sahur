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

@app.post("/coords")
async def receive_coords(request: Request):
    data = await request.json()
    print("Received coordinates from frontend:", data)
    return {"message": "Coordinates received", "coords": data} 