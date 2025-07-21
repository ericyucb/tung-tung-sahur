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

        # 2. Download video from S3 to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            local_video_path = os.path.join(tmpdir, file.filename)
            with open(local_video_path, 'wb') as f:
                s3_client.download_fileobj(AWS_S3_BUCKET_NAME, file.filename, f)

            # 3. Run inference (replace with your actual script/command)
            # Example: python inference.py --input <video> --output <csv>
            output_csv = os.path.splitext(file.filename)[0] + '_result.csv'
            local_csv_path = os.path.join(tmpdir, output_csv)
            try:
                subprocess.run([
                    'python', 'inference.py',
                    '--input', local_video_path,
                    '--output', local_csv_path
                ], check=True)
            except subprocess.CalledProcessError as e:
                return JSONResponse(status_code=500, content={"error": f"Inference failed: {e}"})

            # 4. Upload result CSV to S3
            with open(local_csv_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=AWS_S3_BUCKET_NAME,
                    Key=output_csv,
                    Body=f,
                    ContentType='text/csv'
                )

        return {
            "message": "Upload and inference successful",
            "video_filename": file.filename,
            "csv_s3_key": output_csv
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)}) 