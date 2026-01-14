"""
Camera Test Script

This script tests the connection to FLIR Boson thermal cameras
and displays basic information about the camera setup.

Usage:
    python test_camera.py [--dual]

Author: [Your Name]
Date: January 2026
"""

import argparse
import sys
import time
from typing import Optional

from wrapper_boson import BosonWithTelemetry


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test thermal camera connection")
    parser.add_argument('--dual', action='store_true',
                       help='Test dual camera setup')
    return parser.parse_args()


def test_single_camera() -> bool:
    """
    Test connection to a single thermal camera.
    
    Returns:
        True if test passed, False otherwise
    """
    print("Testing single camera connection...")
    
    try:
        # Initialize camera
        camera = BosonWithTelemetry()
        print("✓ Camera initialized successfully")
        
        # Test image capture
        print("Testing image capture...")
        image, timestamp, frame_number, telemetry = camera.get_next_image()
        
        print(f"✓ Image captured successfully")
        print(f"  Image shape: {image.shape}")
        print(f"  Timestamp: {timestamp:.3f}")
        print(f"  Frame number: {frame_number}")
        print(f"  Telemetry shape: {telemetry.shape}")
        
        # Test a few more captures to check stability
        print("Testing capture stability (5 frames)...")
        for i in range(5):
            image, timestamp, frame_number, _ = camera.get_next_image()
            print(f"  Frame {i+1}: {frame_number} @ {timestamp:.3f}")
            time.sleep(0.1)
        
        print("✓ Camera capture stability test passed")
        
        # Cleanup
        camera.stop()
        camera.close()
        print("✓ Camera closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False


def test_dual_camera() -> bool:
    """
    Test connection to dual thermal cameras.
    
    Returns:
        True if test passed, False otherwise
    """
    print("Testing dual camera connection...")
    
    camera_a = None
    camera_b = None
    
    try:
        # Initialize camera A
        print("Initializing camera A (device 1, COM4)...")
        camera_a = BosonWithTelemetry(device=1, port="COM4")
        print("✓ Camera A initialized successfully")
        
        # Initialize camera B
        print("Initializing camera B (device 2, COM6)...")
        camera_b = BosonWithTelemetry(device=2, port="COM6")
        print("✓ Camera B initialized successfully")
        
        # Test image capture from both cameras
        print("Testing simultaneous capture...")
        
        for i in range(3):
            # Capture from camera A
            image_a, timestamp_a, frame_a, _ = camera_a.get_next_image()
            
            # Capture from camera B
            image_b, timestamp_b, frame_b, _ = camera_b.get_next_image()
            
            time_diff = abs(timestamp_a - timestamp_b) * 1000  # ms
            
            print(f"  Capture {i+1}:")
            print(f"    Camera A: Frame {frame_a} @ {timestamp_a:.3f}")
            print(f"    Camera B: Frame {frame_b} @ {timestamp_b:.3f}")
            print(f"    Time difference: {time_diff:.1f} ms")
            
            time.sleep(0.2)
        
        print("✓ Dual camera capture test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Dual camera test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if camera_a:
            try:
                camera_a.stop()
                camera_a.close()
                print("✓ Camera A closed successfully")
            except Exception as e:
                print(f"Warning: Error closing camera A: {e}")
        
        if camera_b:
            try:
                camera_b.stop()
                camera_b.close()
                print("✓ Camera B closed successfully")
            except Exception as e:
                print(f"Warning: Error closing camera B: {e}")


def display_system_info() -> None:
    """Display system information relevant to camera operation."""
    print("\nSystem Information:")
    print("-" * 30)
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("OpenCV: Not installed")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    try:
        import flirpy
        print("flirpy: Available")
    except ImportError:
        print("flirpy: Not installed")
    
    print(f"Python version: {sys.version}")


def main() -> None:
    """Main test function."""
    args = parse_args()
    
    print("Thermal Camera Connection Test")
    print("=" * 50)
    
    # Display system info
    display_system_info()
    
    print("\n" + "=" * 50)
    
    # Run appropriate test
    if args.dual:
        success = test_dual_camera()
    else:
        success = test_single_camera()
    
    # Display results
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED")
        print("\nYour camera setup is working correctly!")
        if args.dual:
            print("You can now use record_dual_thermal_video.py")
        else:
            print("You can now use record_thermal_video.py")
    else:
        print("✗ TESTS FAILED")
        print("\nTroubleshooting tips:")
        print("1. Check that cameras are connected via USB")
        print("2. Verify FLIR drivers are installed")
        print("3. Make sure no other applications are using the cameras")
        print("4. For dual setup, check COM port assignments")
        sys.exit(1)


if __name__ == "__main__":
    main()