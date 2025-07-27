import os
import sys
import cv2
import argparse

def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file."""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
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
        cv2.imwrite(output_path, frame)
        
        # Release the video capture object
        cap.release()
        
        print(f"First frame extracted and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract first frame from video")
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, required=True, help='Output image file path')
    
    args = parser.parse_args()
    
    success = extract_first_frame(args.input, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()