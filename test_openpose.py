#!/usr/bin/env python3
"""
Test script for OpenPose integration with SAM2 video processing.
This script demonstrates how to process a video with SAM2 and then run OpenPose on the masked frames.
"""

import os
import sys
import tempfile
import subprocess
import json
import numpy as np
from PIL import Image

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_openpose_integration():
    """Test the OpenPose integration with a sample video."""
    
    print("🧪 Testing OpenPose Integration with SAM2")
    print("=" * 50)
    
    # Check if we have a test video in the static directory
    static_dir = os.path.join(os.path.dirname(__file__), 'backend', 'static')
    test_videos = [f for f in os.listdir(static_dir) if f.endswith('.mov') or f.endswith('.mp4')]
    
    if not test_videos:
        print("❌ No test videos found in static directory")
        print("Please upload a video first using the web interface")
        return False
    
    # Use the first available video
    test_video = test_videos[0]
    video_path = os.path.join(static_dir, test_video)
    
    print(f"📹 Using test video: {test_video}")
    
    # Create sample points (you would normally get these from the frontend)
    sample_points = [
        {"x": 500, "y": 300},
        {"x": 520, "y": 280},
        {"x": 480, "y": 320}
    ]
    
    print(f"🎯 Sample points: {sample_points}")
    
    # Test the full pipeline
    try:
        # Run the inference script with video processing
        output_video = f"{os.path.splitext(test_video)[0]}_masked.mp4"
        output_path = os.path.join(static_dir, output_video)
        
        print(f"🔄 Running SAM2 + OpenPose pipeline...")
        
        # Convert points to the format expected by the inference script
        points_np = np.array([[p['x'], p['y']] for p in sample_points], dtype=np.float32)
        
        # Run the inference script
        result = subprocess.run([
            'python3', os.path.join('backend', 'inference.py'),
            '--video', video_path,
            '--points', json.dumps(points_np.tolist()),
            '--video_output', output_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ SAM2 + OpenPose processing completed successfully!")
            print(f"📁 Output video: {output_path}")
            
            if os.path.exists(output_path):
                print("✅ Output video file exists!")
                return True
            else:
                print("❌ Output video file not found")
                return False
        else:
            print(f"❌ Processing failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False

def test_openpose_only():
    """Test OpenPose processing on a regular video (without SAM2 masking)."""
    
    print("\n🧪 Testing OpenPose Only Processing")
    print("=" * 50)
    
    # Check if we have a test video
    static_dir = os.path.join(os.path.dirname(__file__), 'backend', 'static')
    test_videos = [f for f in os.listdir(static_dir) if f.endswith('.mov') or f.endswith('.mp4')]
    
    if not test_videos:
        print("❌ No test videos found")
        return False
    
    test_video = test_videos[0]
    video_path = os.path.join(static_dir, test_video)
    
    print(f"📹 Using test video: {test_video}")
    
    try:
        # Extract frames from video
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = f"/tmp/{video_id}_test_frames"
        os.makedirs(frame_dir, exist_ok=True)
        
        print(f"🔄 Extracting frames to: {frame_dir}")
        
        # Extract frames using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-vf', 'fps=30',
            '-frame_pts', '1',
            os.path.join(frame_dir, '%05d.jpg')
        ], check=True)
        
        # Test OpenPose processing
        output_video_path = os.path.join(static_dir, f"{video_id}_openpose_test.mp4")
        openpose_json_dir = f"/tmp/{video_id}_openpose_test_json"
        
        # Import and run OpenPose function
        from backend.inference import run_openpose_on_masked_frames
        
        print("🔄 Running OpenPose on frames...")
        success = run_openpose_on_masked_frames(
            frame_dir, 
            openpose_json_dir, 
            output_video_path
        )
        
        if success and os.path.exists(output_video_path):
            print("✅ OpenPose processing completed successfully!")
            print(f"📁 Output video: {output_video_path}")
            return True
        else:
            print("❌ OpenPose processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during OpenPose processing: {e}")
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        shutil.rmtree(openpose_json_dir, ignore_errors=True)

if __name__ == "__main__":
    print("🚀 Starting OpenPose Integration Tests")
    print("=" * 60)
    
    # Test 1: Full SAM2 + OpenPose pipeline
    print("\n1️⃣ Testing Full SAM2 + OpenPose Pipeline")
    test1_success = test_openpose_integration()
    
    # Test 2: OpenPose only
    print("\n2️⃣ Testing OpenPose Only Processing")
    test2_success = test_openpose_only()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   SAM2 + OpenPose Pipeline: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   OpenPose Only: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success or test2_success:
        print("\n🎉 OpenPose integration is working!")
        print("💡 You can now use the web interface to process videos with landmarks.")
    else:
        print("\n⚠️  OpenPose integration needs attention.")
        print("💡 Check that OpenPose is properly installed and configured.")
    
    print("=" * 60) 