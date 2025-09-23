#!/usr/bin/env python3
"""
Test script for video generation functionality
"""

import json
import os
from pathlib import Path

def test_video_generation():
    """Test the video generation functionality with Delft data"""
    
    # Check if required data files exist
    delft_pos_path = "data/delft/node_positions.json"
    delft_output_path = "outputs/delft_directions"
    
    if not os.path.exists(delft_pos_path):
        print(f"❌ Missing required file: {delft_pos_path}")
        return False
    
    if not os.path.exists(delft_output_path):
        print(f"❌ Missing required directory: {delft_output_path}")
        print("Please run a Delft simulation first to generate output data")
        return False
    
    try:
        # Import the dashboard class
        from network_dashboard import NetworkDashboard, SELENIUM_AVAILABLE
        
        if not SELENIUM_AVAILABLE:
            print("⚠️  Video generation dependencies not installed")
            print("Install with: pip install -r video_requirements.txt")
            return False
        
        # Load node positions
        with open(delft_pos_path, 'r') as f:
            pos = {str(k): tuple(v) for k, v in json.load(f).items()}
        
        print("✅ Node positions loaded successfully")
        print(f"   Found {len(pos)} nodes")
        
        # Initialize dashboard
        dashboard = NetworkDashboard(delft_output_path, pos)
        print("✅ Dashboard initialized successfully")
        print(f"   Max time steps: {dashboard.max_time}")
        
        # Test video generation (just first 5 frames for quick test)
        print("\n🎬 Testing video generation...")
        
        def progress_callback(current, total):
            print(f"   Frame {current}/{total}")
        
        video_path = dashboard.generate_video(
            start_time=0,
            end_time=5,  # Just 5 frames for testing
            variable='density',
            fps=2,
            format='gif',  # GIF is usually smaller for testing
            width=800,
            height=600,
            progress_callback=progress_callback
        )
        
        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"✅ Video generated successfully!")
            print(f"   Path: {video_path}")
            print(f"   Size: {file_size:.2f} MB")
            return True
        else:
            print("❌ Video generation failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Video Generation Functionality")
    print("=" * 50)
    
    success = test_video_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! Video generation is working correctly.")
    else:
        print("❌ Tests failed. Please check the error messages above.")
