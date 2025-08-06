# OpenPose Integration with SAM2 Video Processing

This document explains how OpenPose has been integrated with the SAM2 video processing pipeline to create videos with pose landmarks.

## Overview

The system now supports running OpenPose on masked videos to create videos with pose landmarks. This is useful for analyzing human movement patterns in medical rehabilitation videos.

## Workflow

1. **Video Upload**: User uploads a video through the web interface
2. **Frame Extraction**: First frame is extracted and displayed for annotation
3. **Point Selection**: User clicks points on the first frame to define the target object
4. **SAM2 Segmentation**: SAM2 creates masks for the target object throughout the video
5. **Masked Video Creation**: A masked video is created showing only the segmented object
6. **OpenPose Processing**: OpenPose runs on the masked frames to detect pose landmarks
7. **Final Video**: A video with pose landmarks is returned

## API Endpoints

### 1. Full Pipeline (SAM2 + OpenPose)
```
POST /process_video
```
Processes a video with both SAM2 segmentation and OpenPose landmark detection.

**Request Body:**
```json
{
  "video_filename": "example.mov",
  "points": [
    {"x": 500, "y": 300},
    {"x": 520, "y": 280},
    {"x": 480, "y": 320}
  ]
}
```

**Response:**
```json
{
  "message": "Full video processing completed with SAM2 and OpenPose",
  "masked_video_url": "/video/example_masked.mp4",
  "has_landmarks": true
}
```

### 2. OpenPose Only
```
POST /process_video_openpose
```
Processes a video with OpenPose only (no SAM2 masking).

**Request Body:**
```json
{
  "video_filename": "example.mov"
}
```

**Response:**
```json
{
  "message": "OpenPose processing completed",
  "openpose_video_url": "/video/example_openpose.mp4",
  "has_landmarks": true
}
```

## Implementation Details

### OpenPose Function
The `run_openpose_on_masked_frames()` function in `backend/inference.py` handles OpenPose processing:

```python
def run_openpose_on_masked_frames(masked_frame_dir, output_json_dir, output_video_path):
    """Run OpenPose on masked frames to create video with landmarks."""
    # OpenPose command based on allcode.py pattern
    openpose_cmd = [
        openpose_bin,
        "--image_dir", masked_frame_dir,
        "--write_json", output_json_dir,
        "--write_video", output_video_path,
        "--display", "0",
        "--model_pose", "BODY_25",
        "--net_resolution", "-1x368",
        "--write_video_fps", "30"
    ]
```

### Integration with SAM2 Pipeline
The `process_video_with_sam2()` function now includes OpenPose processing:

1. SAM2 creates masked frames
2. OpenPose runs on the masked frames
3. The OpenPose video with landmarks replaces the masked video
4. Temporary files are cleaned up

## OpenPose Configuration

- **Model**: BODY_25 (25 keypoints)
- **Resolution**: Adaptive (-1x368)
- **FPS**: 30
- **Output**: JSON keypoints + video with landmarks

## File Structure

```
backend/
├── inference.py          # Contains OpenPose integration
├── main.py              # API endpoints for OpenPose
├── static/              # Output videos with landmarks
└── allcode.py           # Reference implementation
```

## Testing

Run the test script to verify OpenPose integration:

```bash
python test_openpose.py
```

This will test both the full SAM2 + OpenPose pipeline and OpenPose-only processing.

## Requirements

- OpenPose binary must be available in the system
- Common locations checked:
  - `/content/openpose/build/examples/openpose/openpose.bin`
  - `./build/examples/openpose/openpose.bin`
  - `/usr/local/openpose/build/examples/openpose/openpose.bin`
  - `/opt/openpose/build/examples/openpose/openpose.bin`

## Error Handling

- If OpenPose is not available, the system falls back to returning the masked video without landmarks
- Temporary files are cleaned up even if processing fails
- Detailed error messages are logged for debugging

## Example Usage

1. Upload a video through the web interface
2. Click points on the first frame to define the target object
3. Submit the points to start processing
4. The system will return a video with pose landmarks
5. The video can be downloaded and viewed in the browser

## Benefits

- **Focused Analysis**: OpenPose runs on masked frames, reducing noise from background
- **Medical Applications**: Useful for analyzing patient movement patterns
- **Automated Processing**: No manual pose annotation required
- **Standardized Output**: Consistent BODY_25 keypoint format

## Future Enhancements

- Support for different OpenPose models (COCO, MPI, etc.)
- Keypoint data export in various formats
- Real-time processing capabilities
- Integration with additional pose analysis tools 