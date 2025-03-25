#!/usr/bin/env python3
"""
Dart Detector Configuration Utility

This script provides utilities for configuring the enhanced dart detector
including noise reduction, calibration, and angle detection optimization.

Usage:
  python dart_detector_config.py --calibrate  # Run interactive calibration
  python dart_detector_config.py --optimize   # Run auto-optimization
  python dart_detector_config.py --reset      # Reset to default configuration
  python dart_detector_config.py --test       # Run accuracy test
  python dart_detector_config.py --noise      # Optimize noise reduction
  python dart_detector_config.py --angle      # Optimize angle detection
"""

import argparse
import cv2
import numpy as np
import json
import os
import time
import sys
import random
from enhanced_dart_detector import EnhancedDartDetector

def run_interactive_calibration():
    """Run the interactive calibration mode"""
    print("Starting Interactive Calibration Mode")
    print("=====================================")
    print("This mode allows you to fine-tune detector parameters in real-time.")
    print("\nInstructions:")
    print("1. Use the trackbars to adjust parameters")
    print("2. Press 'b' to capture the background (do this with no dart in view)")
    print("3. Test detection with darts at different board positions")
    print("4. Press 's' to save your configuration")
    print("5. Press 'q' to exit")
    print("\nStarting calibration...")
    
    detector = EnhancedDartDetector()
    detector.calibrate_interactively()
    
    print("Calibration completed.")

def test_accuracy():
    """Run accuracy test to measure repeatability"""
    print("Starting Accuracy Testing Mode")
    print("=============================")
    print("This mode tests the repeatability of the detector.")
    print("\nInstructions:")
    print("1. Place a dart at a fixed position on the board")
    print("2. The system will record multiple measurements")
    print("3. Statistics on position and angle repeatability will be displayed")
    print("\nStarting test...")
    
    # Create detector with saved configuration
    detector = EnhancedDartDetector()
    detector.load_config()
    
    # Open cameras
    cap1 = cv2.VideoCapture(detector.cam_index1)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, detector.frame_width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.frame_height)
    
    cap2 = cv2.VideoCapture(detector.cam_index2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, detector.frame_width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.frame_height)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both cameras.")
        return
    
    # Initialize background subtraction
    print("Capturing background (ensure no dart is visible)...")
    for _ in range(30):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 and ret2:
            frame1_rot = cv2.rotate(frame1, cv2.ROTATE_180)
            frame2_rot = cv2.rotate(frame2, cv2.ROTATE_180)
            
            roi1 = frame1_rot[detector.cam1_roi_top:detector.cam1_roi_bottom, :]
            roi2 = frame2_rot[detector.cam2_roi_top:detector.cam2_roi_bottom, :]
            
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            
            detector.bg_subtractor1.apply(gray1, learningRate=0.1)
            detector.bg_subtractor2.apply(gray2, learningRate=0.1)
            
            cv2.waitKey(1)
    
    # Ask the user to place the dart
    print("\nPlace a dart on the board at a fixed position.")
    print("Press Enter when ready to start collecting measurements.")
    input()
    
    # Collect measurements
    num_samples = 50
    print(f"\nCollecting {num_samples} measurements. Please keep the dart still...")
    
    positions = []
    cam1_angles = []
    cam2_angles = []
    
    for i in range(num_samples):
        # Read frames
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error: Failed to grab frames.")
            break
        
        # Process frames
        proc_frame1, mask1 = detector.process_camera1_frame(frame1)
        proc_frame2, mask2 = detector.process_camera2_frame(frame2)
        
        # Calculate intersection
        if detector.cam1_vector is not None and detector.cam2_vector is not None:
            tip = detector.compute_intersection()
            if tip is not None:
                positions.append(tip)
            
            if detector.cam1_angle is not None:
                cam1_angles.append(detector.cam1_angle)
            
            if detector.cam2_angle is not None:
                cam2_angles.append(detector.cam2_angle)
        
        # Display progress
        sys.stdout.write(f"\rCollecting measurements: {i+1}/{num_samples}")
        sys.stdout.flush()
        
        # Display what's being measured
        board_vis = detector.create_board_visualization()
        cv2.imshow("Measurement Visualization", board_vis)
        cv2.imshow("Camera 1", proc_frame1)
        cv2.imshow("Camera 2", proc_frame2)
        
        cv2.waitKey(100)  # Small delay between measurements
    
    print("\n\nMeasurement collection completed.")
    
    # Calculate statistics
    if len(positions) > 0:
        x_values = [p[0] for p in positions]
        y_values = [p[1] for p in positions]
        
        # Position statistics
        mean_x = np.mean(x_values)
        mean_y = np.mean(y_values)
        std_x = np.std(x_values)
        std_y = np.std(y_values)
        
        # Distance from each point to mean
        distances = [np.sqrt((x - mean_x)**2 + (y - mean_y)**2) for x, y in zip(x_values, y_values)]
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Angle statistics
        cam1_angle_std = np.std(cam1_angles) if len(cam1_angles) > 0 else float('nan')
        cam2_angle_std = np.std(cam2_angles) if len(cam2_angles) > 0 else float('nan')
        
        # Print results
        print("\nPosition Repeatability Statistics:")
        print(f"Number of valid measurements: {len(positions)}/{num_samples}")
        print(f"Mean position: ({mean_x:.2f}, {mean_y:.2f}) mm")
        print(f"Standard deviation: X={std_x:.2f} mm, Y={std_y:.2f} mm")
        print(f"Mean distance from center: {mean_distance:.2f} mm")
        print(f"Maximum deviation: {max_distance:.2f} mm")
        
        print("\nAngle Repeatability Statistics:")
        print(f"Camera 1 angle std dev: {cam1_angle_std:.2f}°")
        print(f"Camera 2 angle std dev: {cam2_angle_std:.2f}°")
        
        # Save results to file
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": num_samples,
            "valid_measurements": len(positions),
            "mean_position": [float(mean_x), float(mean_y)],
            "std_dev": [float(std_x), float(std_y)],
            "mean_distance": float(mean_distance),
            "max_distance": float(max_distance),
            "cam1_angle_std": float(cam1_angle_std) if not np.isnan(cam1_angle_std) else None,
            "cam2_angle_std": float(cam2_angle_std) if not np.isnan(cam2_angle_std) else None
        }
        
        with open("accuracy_test_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        print("\nResults saved to accuracy_test_results.json")
    else:
        print("No valid measurements collected. Check camera setup and dart visibility.")
    
    # Clean up
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
