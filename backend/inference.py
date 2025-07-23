import cv2
import sys
import json

def extract_first_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read video: {video_path}")
        sys.exit(1)
    cv2.imwrite(output_image_path, frame)
    cap.release()
    print(f"First frame saved to {output_image_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract first frame or print coordinates.")
    parser.add_argument('--input', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output image file path')
    parser.add_argument('--coords', type=str, help='JSON string or file path with coordinates')
    args = parser.parse_args()

    if args.input and args.output:
        extract_first_frame(args.input, args.output)
    elif args.coords:
        # Accepts either a JSON string or a file path
        try:
            if args.coords.endswith('.json'):
                with open(args.coords, 'r') as f:
                    coords = json.load(f)
            else:
                coords = json.loads(args.coords)
            print(f"Received coordinates: {coords}")
        except Exception as e:
            print(f"Failed to parse coordinates: {e}")
            sys.exit(1)
    else:
        print("No valid arguments provided.")
        sys.exit(1)