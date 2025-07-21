# AWS Video Upload Template

This template provides a basic setup for uploading videos from a Next.js frontend to a FastAPI backend running on an EC2 instance, which then stores the videos in an S3 bucket.

## Structure

- `frontend/`: Next.js app with a video upload UI
- `backend/`: FastAPI app to receive uploads and send to S3

## Setup Instructions

### 1. Backend (FastAPI)
- Install dependencies: `pip install -r requirements.txt`
- Set environment variables:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_S3_BUCKET_NAME`
  - `AWS_REGION`
- Run the server: `uvicorn main:app --host 0.0.0.0 --port 8000`

### 2. Frontend (Next.js)
- Install dependencies: `npm install`
- Set the backend API URL in `.env.local`:
  - `NEXT_PUBLIC_API_URL=http://<EC2_PUBLIC_IP>:8000`
- Run the app: `npm run dev`

## Notes
- All video uploads go through the backend; the frontend does not upload directly to S3.
- The backend is ready to be containerized for EC2 deployment.
- Extend the backend to trigger AI inference as needed. # tung-tung-sahur
