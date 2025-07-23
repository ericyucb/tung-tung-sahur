# AWS Video Upload Template

This template provides a basic setup for uploading videos from a Next.js frontend to a FastAPI backend running on an EC2 instance, which then stores the videos in an S3 bucket.

## Structure

- `frontend/`: Next.js app with a video upload UI
- `backend/`: FastAPI app to receive uploads and send to S3

## Setup Instructions

### 1. Backend (FastAPI)
- Create and activate a Python virtual environment in the project root:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r backend/requirements.txt
  ```
- Create a `.env` file in the `backend/` directory with the following variables:
  ```
  AWS_ACCESS_KEY_ID=your-access-key
  AWS_SECRET_ACCESS_KEY=your-secret-key
  AWS_S3_BUCKET_NAME=your-bucket-name
  AWS_REGION=your-region
  ```
- Run the server:
  ```bash
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
  ```

### 2. Frontend (Next.js)
- Install dependencies:
  ```bash
  cd frontend
  npm install
  ```
- Create a `.env.local` file in the `frontend/` directory with the following variable:
  ```
  NEXT_PUBLIC_API_URL=http://localhost:8000
  ```
  (Change the URL if your backend is running elsewhere)
- Run the app:
  ```bash
  npm run dev
  ```

## Usage
- Open the frontend in your browser at `http://localhost:3000`.
- Use the UI to upload a video file.
- The backend will receive the file and upload it directly to your configured AWS S3 bucket.
- You will receive a success message when the upload is complete.

## Notes
- The backend **no longer runs inference** or processes the video after upload. It simply stores the uploaded file in S3.
- All video uploads go through the backend; the frontend does not upload directly to S3.
- The backend is ready to be containerized for EC2 deployment.
- Make sure your AWS credentials and bucket permissions are set correctly.
