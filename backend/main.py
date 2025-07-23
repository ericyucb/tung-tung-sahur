from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import boto3
import os
from dotenv import load_dotenv
import tempfile
import subprocess

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

        # 2. Return success response (skip inference)
        return {
            "message": "Upload successful",
            "video_filename": file.filename,
            "s3_key": file.filename
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)}) 