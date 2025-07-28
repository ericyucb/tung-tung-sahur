#!/usr/bin/env python3
"""
Test script to verify SAM2 integration is working.
"""

import requests
import json
import base64
import os

# Test the backend endpoints
BASE_URL = "http://3.239.9.0:8000"

def test_backend_health():
    """Test if the backend is responding."""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Backend health check: {response.status_code}")
        return True
    except Exception as e:
        print(f"Backend health check failed: {e}")
        return False

def test_segment_endpoint():
    """Test the segment endpoint with dummy data."""
    try:
        # Create a simple test image (1x1 pixel)
        test_data = {
            "points": [{"x": 100, "y": 100}],
            "video_filename": "test_video.mp4"
        }
        
        response = requests.post(
            f"{BASE_URL}/segment",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Segment endpoint test: {response.status_code}")
        if response.status_code == 200:
            print("✅ Segment endpoint is working!")
        else:
            print(f"❌ Segment endpoint failed: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Segment endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing SAM2 Backend Integration...")
    print("=" * 40)
    
    # Test backend health
    if test_backend_health():
        print("✅ Backend is responding")
    else:
        print("❌ Backend is not responding")
        exit(1)
    
    # Test segment endpoint
    if test_segment_endpoint():
        print("✅ SAM2 integration appears to be working")
    else:
        print("❌ SAM2 integration may have issues")
    
    print("\n🎉 Test completed!") 