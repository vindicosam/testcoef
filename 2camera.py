#!/usr/bin/env python3
"""
Enhanced Dual Camera Dart Detector with Configuration Utilities

This script implements a dart detection system using two cameras to track
dart position and lean angle with high accuracy and noise resistance.

Usage:
  python combined_dart_detector.py             # Run the detector
  python combined_dart_detector.py --calibrate # Run interactive calibration
  python combined_dart_detector.py --test      # Run accuracy test
  python combined_dart_detector.py --optimize  # Run auto-optimization
  python combined_dart_detector.py --reset     # Reset to default configuration
"""

import cv2
import numpy as np
import math
import sys
import signal
import time
import json
import os
import argparse
import random
from collections import deque

class EnhancedDartDetector:
    def __init__(self, cam_index1=0, cam_index2=2, config=None):
        # Default configuration
        self.config = {
            # Camera settings
            "frame_width": 640,
            "frame_height": 480,
            
            # Camera positions (mm)
            "camera1_position": (0, 350),    # Front camera fixed position
            "camera2_position": (-650, 0),   # Side camera fixed position
            
            # ROI settings for Camera 1
            "cam1_board_plane_y": 198,
            "cam1_roi_range": 30,
            
            # ROI settings for Camera 2
            "cam2_board_plane_y": 199,
            "cam2_roi_range": 30,
            
            # Calibration parameters
            "camera1_pixel_to_mm_factor": -0.782,  # Slope in mm/pixel
            "camera1_pixel_offset": 226.8,         # Board x when pixel_x = 0
            "camera2_pixel_to_mm_factor": -0.628,  # Slope in mm/pixel
            "camera2_pixel_offset": 192.8,         # Board y when pixel_x = 0
            
            # Background subtraction
            "history": 100,
            "var_threshold": 30,
            "learning_rate": 0.01,
            
            # Contour filtering
            "min_contour_area": 15,
            "max_contour_area": 2000,
            
            # Angle detection
            "angle_search_depth": 25,
            "angle_search_width": 40,
            "min_points_for_angle": 8,
            
            # Temporal filtering
            "position_buffer_size": 5,
            "angle_buffer_size": 7,
            
            # Board settings
            "board_radius": 170,
            "board_segments": {
                4: (90, 50),
                5: (-20, 103),
                16: (90, -50),
                17: (20, -100)
            },
            
            # Calibration points
            "calibration_points": {
                (0, 0): (290, 307),
                (-171, 0): (506, 307),
                (171, 0): (68, 307),
                (0, 171): (290, 34),
                (0, -171): (290, 578),
                (90, 50): (151, 249),
                (-20, 103): (327, 131),
                (20, -100): (277, 459),
                (90, -50): (359, 406)
            }
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2
        
        # Initialize camera and board settings from config
        self._initialize_settings()
        
        # Initialize background subtractors with tuned parameters
        self._initialize_background_subtractors()
        
        # Initialize result buffers for temporal filtering
        self._initialize_buffers()
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Running flag
        self.running = True
        
        # Load the dartboard image if available
        self.board_image = None
        try:
            self.board_image = cv2.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        except:
            pass
            
        # State variables for detection
        self.is_detecting = False
        self.last_detection_time = 0
        self.detection_timeout = 3.0  # seconds
        
    def _initialize_settings(self):
        """Initialize settings from config"""
        # Camera settings
        self.frame_width = self.config["frame_width"]
        self.frame_height = self.config["frame_height"]
        
        # Static camera positions
        self.camera1_position = self.config["camera1_position"]
        self.camera2_position = self.config["camera2_position"]
        
        # ROI settings
        self.cam1_board_plane_y = self.config["cam1_board_plane_y"]
        self.cam1_roi_range = self.config["cam1_roi_range"]
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        self.cam2_board_plane_y = self.config["cam2_board_plane_y"]
        self.cam2_roi_range = self.config["cam2_roi_range"]
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        
        # Calibration parameters
        self.camera1_pixel_to_mm_factor = self.config["camera1_pixel_to_mm_factor"]
        self.camera1_pixel_offset = self.config["camera1_pixel_offset"]
        self.camera2_pixel_to_mm_factor = self.config["camera2_pixel_to_mm_factor"]
        self.camera2_pixel_offset = self.config["camera2_pixel_offset"]
        
        # Board settings
        self.board_radius = self.config["board_radius"]
        self.board_segments = self.config["board_segments"]
        self.calibration_points = self.config["calibration_points"]
    
    def _initialize_background_subtractors(self):
        """Initialize background subtraction with configured parameters"""
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
            history=self.config["history"], 
            varThreshold=self.config["var_threshold"], 
            detectShadows=False
        )
        
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
            history=self.config["history"], 
            varThreshold=self.config["var_threshold"], 
            detectShadows=False
        )
    
    def _initialize_buffers(self):
        """Initialize buffers for temporal filtering"""
        # Position buffers
        self.cam1_vector_buffer = deque(maxlen=self.config["position_buffer_size"])
        self.cam2_vector_buffer = deque(maxlen=self.config["position_buffer_size"])
        self.final_tip_buffer = deque(maxlen=self.config["position_buffer_size"])
        
        # Angle buffers
        self.cam1_angle_buffer = deque(maxlen=self.config["angle_buffer_size"])
        self.cam2_angle_buffer = deque(maxlen=self.config["angle_buffer_size"])
        
        # Current values
        self.cam1_vector = None
        self.cam2_vector = None
        self.final_tip = None
        self.cam1_angle = None
        self.cam2_angle = None
        
    def signal_handler(self, signum, frame):
        """Handle CTRL+C"""
        self.running = False
        print("\nShutting down...")
        sys.exit(0)
    
    def detect_dart_tip(self, mask, min_area=None, max_area=None):
        """
        Detect the dart tip position from binary mask with improved noise filtering
        
        Args:
            mask: Binary mask containing dart
            min_area: Minimum contour area (if None, uses config)
            max_area: Maximum contour area (if None, uses config)
            
        Returns:
            tuple: (tip_point, contours) or (None, []) if no valid dart tip found
        """
        if min_area is None:
            min_area = self.config["min_contour_area"]
        if max_area is None:
            max_area = self.config["max_contour_area"]
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, []
        
        # Find the highest point (lowest y-value) across all valid contours
        tip_point = None
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:  # Filter out noise and overly large objects
                continue
                
            valid_contours.append(contour)
            
            # Get the highest point in this contour (lowest y-value)
            # This is more reliable for finding the dart tip than using the whole contour
            min_y_point = None
            min_y = float('inf')
            
            for point in contour:
                x, y = point[0]
                if y < min_y:
                    min_y = y
                    min_y_point = (x, y)
            
            if min_y_point is not None:
                if tip_point is None or min_y_point[1] < tip_point[1]:
                    tip_point = min_y_point
        
        return tip_point, valid_contours
    
    def measure_tip_angle(self, mask, tip_point):
        """
        Measure the angle of the dart tip with improved accuracy.
        Uses a modified linear regression approach that better handles dart shaft shape.
        
        Args:
            mask: Binary mask containing dart
            tip_point: Detected tip coordinates (x,y)
            
        Returns:
            tuple: (angle, lean, points) or None if angle couldn't be calculated
        """
        if tip_point is None:
            return None
            
        tip_x, tip_y = tip_point
        
        # Define search parameters
        search_depth = self.config["angle_search_depth"]
        search_width = self.config["angle_search_width"]
        min_points = self.config["min_points_for_angle"]
        
        # Define region to search for the dart shaft
        min_x = max(0, tip_x - search_width)
        max_x = min(mask.shape[1] - 1, tip_x + search_width)
        max_y = min(mask.shape[0] - 1, tip_y + search_depth)
        
        # Find all white pixels in the search area
        points_below = []
        for y in range(tip_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if mask[y, x] > 0:  # White pixel
                    points_below.append((x, y))
        
        if len(points_below) < min_points:  # Need enough points for a good fit
            return None
            
        # Use RANSAC for more robust angle estimation
        # This helps ignore outlier points that may come from noise
        best_angle = None
        best_inliers = 0
        best_points = []
        
        for _ in range(10):  # Try several random samples
            if len(points_below) < 2:
                continue
                
            # Randomly select two points
            indices = np.random.choice(len(points_below), 2, replace=False)
            p1 = points_below[indices[0]]
            p2 = points_below[indices[1]]
            
            # Skip if points are too close together
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 5:
                continue
                
            # Calculate slope
            if dx == 0:  # Vertical line
                slope = float('inf')
                angle = 90  # Vertical
            else:
                slope = dy / dx
                # Convert slope to angle in degrees (relative to vertical)
                angle_from_horizontal = math.degrees(math.atan(slope))
                angle = 90 - angle_from_horizontal
            
            # Count inliers
            inliers = []
            for point in points_below:
                # Distance from point to line defined by p1, p2
                if dx == 0:  # Vertical line
                    dist_to_line = abs(point[0] - p1[0])
                else:
                    a = -slope
                    b = 1
                    c = slope * p1[0] - p1[1]
                    dist_to_line = abs(a*point[0] + b*point[1] + c) / math.sqrt(a*a + b*b)
                
                if dist_to_line < 2:  # Threshold distance for inlier
                    inliers.append(point)
            
            if len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_angle = angle
                best_points = inliers
        
        if best_angle is None:
            # Fall back to simple linear regression if RANSAC fails
            points = np.array(points_below)
            x = points[:, 0]
            y = points[:, 1]
            
            # Calculate slope using least squares
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator == 0:  # Avoid division by zero
                slope = float('inf')
                best_angle = 90  # Vertical line
            else:
                slope = numerator / denominator
                angle_from_horizontal = math.degrees(math.atan(slope))
                best_angle = 90 - angle_from_horizontal
                
            best_points = points_below
        
        # Determine lean direction
        lean = "VERTICAL"
        if best_angle < 85:
            lean = "LEFT"
        elif best_angle > 95:
            lean = "RIGHT"
            
        return best_angle, lean, best_points
    
    def apply_temporal_filtering(self, value, buffer):
        """
        Apply temporal filtering to smooth out measurements
        Using a moving median which is more robust to outliers than a mean
        
        Args:
            value: New value to add to buffer
            buffer: Deque buffer to add value to
            
        Returns:
            float: Filtered value
        """
        if value is None:
            return None
            
        # Add value to buffer
        buffer.append(value)
        
        # Return median value from buffer
        if len(buffer) > 0:
            if isinstance(value, tuple) and len(value) == 2:  # For 2D points
                x_values = [p[0] for p in buffer if p is not None]
                y_values = [p[1] for p in buffer if p is not None]
                
                if len(x_values) > 0 and len(y_values) > 0:
                    x_median = sorted(x_values)[len(x_values) // 2]
                    y_median = sorted(y_values)[len(y_values) // 2]
                    return (x_median, y_median)
                else:
                    return None
            else:  # For scalar values
                values = [v for v in buffer if v is not None]
                if len(values) > 0:
                    return sorted(values)[len(values) // 2]
                else:
                    return None
        else:
            return None
    
    def process_camera1_frame(self, frame):
        """
        Process the frame from Camera 1 with improved dart tip and angle detection
        
        Args:
            frame: Input frame from camera 1
            
        Returns:
            tuple: (processed_frame, mask)
        """
        # Rotate the frame if needed
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        roi_vis = roi.copy()
        
        # Apply background subtraction
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Determine learning rate based on detection state
        learning_rate = -1 if self.is_detecting else self.config["learning_rate"]
        fg_mask = self.bg_subtractor1.apply(gray, learningRate=learning_rate)
        
        # Threshold and clean up the mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Detect dart tip
        tip_point, contours = self.detect_dart_tip(thresh)
        
        # Draw all contours
        if contours:
            cv2.drawContours(roi_vis, contours, -1, (0, 255, 0), 2)
        
        # Draw a line at the board plane y-value
        plane_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi_vis, (0, plane_y), (roi.shape[1], plane_y), (0, 255, 255), 1)
        
        dart_pixel_x = None
        new_cam1_angle = None
        
        # Process if tip detected
        if tip_point is not None:
            # Mark the tip point
            cv2.circle(roi_vis, tip_point, 5, (0, 0, 255), -1)
            
            # Use pixel x position for dartboard mapping
            dart_pixel_x = tip_point[0]
            
            # Measure angle
            angle_info = self.measure_tip_angle(thresh, tip_point)
            
            if angle_info is not None:
                angle, lean, points_below = angle_info
                new_cam1_angle = angle
                
                # Draw all detected points
                for point in points_below:
                    cv2.circle(roi_vis, point, 1, (255, 0, 255), -1)
                
                # Draw the line fitted through these points
                if len(points_below) > 1:
                    # Use the calculated angle to draw the line
                    angle_rad = math.radians(90 - angle)  # Convert to radians from horizontal
                    slope = math.tan(angle_rad)
                    
                    # Calculate line through tip point
                    line_length = 50
                    x1 = tip_point[0] - line_length
                    y1 = int(tip_point[1] + slope * (x1 - tip_point[0]))
                    x2 = tip_point[0] + line_length
                    y2 = int(tip_point[1] + slope * (x2 - tip_point[0]))
                    
                    # Draw line
                    cv2.line(roi_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Apply temporal filtering to angle
                self.cam1_angle = self.apply_temporal_filtering(new_cam1_angle, self.cam1_angle_buffer)
                
                # Display angle info on ROI
                roi_text = f"Angle: {self.cam1_angle:.1f}° ({lean})"
                cv2.putText(roi_vis, roi_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Convert dart_pixel_x to board coordinates
        if dart_pixel_x is not None:
            # Apply calibration for cam1 to get board x-coordinate
            board_x = self.camera1_pixel_to_mm_factor * dart_pixel_x + self.camera1_pixel_offset
            # Store vector info - we know this passes through (board_x, 0) on the board
            new_cam1_vector = (board_x, 0)
            
            # Apply temporal filtering
            self.cam1_vector = self.apply_temporal_filtering(new_cam1_vector, self.cam1_vector_buffer)
            
            # Add calibration debugging info to display
            cv2.putText(roi_vis, f"Board X: {self.cam1_vector[0]:.1f}mm", (10, roi_vis.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None
        
        # Copy ROI visualization back to the frame
        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi_vis
        
        return frame_rot, thresh
    
    def process_camera2_frame(self, frame):
        """
        Process the frame from Camera 2 with improved dart tip and angle detection
        
        Args:
            frame: Input frame from camera 2
            
        Returns:
            tuple: (processed_frame, mask)
        """
        # Rotate the frame if needed
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        roi_vis = roi.copy()
        
        # Apply background subtraction
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Determine learning rate based on detection state
        learning_rate = -1 if self.is_detecting else self.config["learning_rate"]
        fg_mask = self.bg_subtractor2.apply(gray, learningRate=learning_rate)
        
        # Threshold and clean up the mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Detect dart tip
        tip_point, contours = self.detect_dart_tip(thresh)
        
        # Draw all contours
        if contours:
            cv2.drawContours(roi_vis, contours, -1, (0, 255, 0), 2)
        
        # Draw a line at the board plane y-value
        plane_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi_vis, (0, plane_y), (roi.shape[1], plane_y), (0, 255, 255), 1)
        
        dart_pixel_x = None
        new_cam2_angle = None
        
        # Process if tip detected
        if tip_point is not None:
            # Mark the tip point
            cv2.circle(roi_vis, tip_point, 5, (0, 0, 255), -1)
            
            # Use pixel x position for dartboard mapping
            dart_pixel_x = tip_point[0]
            
            # Measure angle
            angle_info = self.measure_tip_angle(thresh, tip_point)
            
            if angle_info is not None:
                angle, lean, points_below = angle_info
                new_cam2_angle = angle
                
                # Draw all detected points
                for point in points_below:
                    cv2.circle(roi_vis, point, 1, (255, 0, 255), -1)
                
                # Draw the line fitted through these points
                if len(points_below) > 1:
                    # Use the calculated angle to draw the line
                    angle_rad = math.radians(90 - angle)  # Convert to radians from horizontal
                    slope = math.tan(angle_rad)
                    
                    # Calculate line through tip point
                    line_length = 50
                    x1 = tip_point[0] - line_length
                    y1 = int(tip_point[1] + slope * (x1 - tip_point[0]))
                    x2 = tip_point[0] + line_length
                    y2 = int(tip_point[1] + slope * (x2 - tip_point[0]))
                    
                    # Draw line
                    cv2.line(roi_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Apply temporal filtering to angle
                self.cam2_angle = self.apply_temporal_filtering(new_cam2_angle, self.cam2_angle_buffer)
                
                # Display angle info on ROI
                roi_text = f"Angle: {self.cam2_angle:.1f}° ({lean})"
                cv2.putText(roi_vis, roi_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Convert dart_pixel_x to board coordinates
        if dart_pixel_x is not None:
            # Apply calibration for cam2 to get board y-coordinate
            board_y = self.camera2_pixel_to_mm_factor * dart_pixel_x + self.camera2_pixel_offset
            # Store vector info - we know this passes through (0, board_y) on the board
            new_cam2_vector = (0, board_y)
            
            # Apply temporal filtering
            self.cam2_vector = self.apply_temporal_filtering(new_cam2_vector, self.cam2_vector_buffer)
            
            # Add calibration debugging info to display
            cv2.putText(roi_vis, f"Board Y: {self.cam2_vector[1]:.1f}mm", (10, roi_vis.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
        
        # Copy ROI visualization back to the frame
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi_vis
        
        return frame_rot, thresh
    
    def compute_line_intersection(self, p1, p2, p3, p4):
        """
        Compute the intersection of two lines.
        p1, p2 define the first line, p3, p4 define the second line.
        Returns the intersection point (x, y) or None if lines are parallel.
        """
        denominator = ((p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0]))
        if abs(denominator) < 1e-10:
            return None
        num_x = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[0] - p4[0])
                 - (p1[0]-p2[0]) * (p3[0]*p4[1] - p3[1]*p4[0]))
        num_y = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[1] - p4[1])
                 - (p1[1]-p2[1]) * (p3[0]*p4[1] - p3[1]*p4[0]))
        x = num_x / denominator
        y = num_y / denominator
        return (x, y)
    
    def compute_intersection(self):
        """
        Compute the intersection of the vectors from both cameras.
        This finds the 3D position of the dart tip.
        
        Returns:
            tuple: (x, y) coordinates of dart tip or None if not available
        """
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
            
        # For cam1, we've determined the board_x value where the vector passes through the board plane
        # Create a ray from camera1 through this point
        cam1_board_x = self.cam1_vector[0]
        cam1_ray_start = self.camera1_position
        cam1_ray_end = (cam1_board_x, 0)  # This is where the ray passes through the board
        
        # For cam2, we've determined the board_y value where the vector passes through the board plane
        # Create a ray from camera2 through this point
        cam2_board_y = self.cam2_vector[1]
        cam2_ray_start = self.camera2_position
        cam2_ray_end = (0, cam2_board_y)  # This is where the ray passes through the board
        
        # Find the intersection of these rays
        intersection = self.compute_line_intersection(
            cam1_ray_start, cam1_ray_end, 
            cam2_ray_start, cam2_ray_end
        )
        
        if intersection is not None:
            # Check if the intersection point is within a reasonable distance from the board
            dist_from_center = math.sqrt(intersection[0]**2 + intersection[1]**2)
            if dist_from_center > self.board_radius * 1.5:  # Allow some margin
                return None
        
        return intersection
    
    def create_board_visualization(self):
        """
        Create a high-quality visualization of the board with dart position and angles
        
        Returns:
            numpy.ndarray: Visualization image
        """
        # Create a canvas for the dartboard visualization
        canvas_size = 800
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background
        
        # Calculate scale factor (px/mm)
        scale_factor = 1.8  # Adjust this to control size
        
        # Center of canvas
        center_x = canvas_size // 2
        center_y = canvas_size // 2
        
        # Function to convert board mm to canvas pixels
        def mm_to_canvas_px(x, y):
            px = int(center_x + x * scale_factor)
            py = int(center_y - y * scale_factor)  # Y inverted in pixel coordinates
            return (px, py)
        
        # Draw board background - display the actual dartboard image if available
        if self.board_image is not None:
            # The dartboard image needs to fill the entire board circle
            board_size = int(self.board_radius * 2 * scale_factor)
            
            # Scale factor multiplier - increase this to make the dartboard larger relative to the boundary
            image_scale_multiplier = 2.75  # Adjust this value to make the dartboard fill the boundary circle
            board_img_size = int(board_size * image_scale_multiplier)
            
            board_resized = cv2.resize(self.board_image, (board_img_size, board_img_size))
            
            # Calculate position to paste the board image (centered)
            board_x = center_x - board_img_size // 2
            board_y = center_y - board_img_size // 2
            
            # Create a circular mask for the board
            mask = np.zeros((board_img_size, board_img_size), dtype=np.uint8)
            cv2.circle(mask, (board_img_size//2, board_img_size//2), board_img_size//2, 255, -1)
            
            # Paste the board image onto the canvas
            # Make sure we don't go out of bounds
            if (board_x >= 0 and board_y >= 0 and 
                board_x + board_img_size <= canvas_size and 
                board_y + board_img_size <= canvas_size):
                canvas_roi = canvas[board_y:board_y+board_img_size, board_x:board_x+board_img_size]
                board_masked = cv2.bitwise_and(board_resized, board_resized, mask=mask)
                canvas_roi[mask > 0] = board_masked[mask > 0]
        
        # Draw coordinate axes
        cv2.line(canvas, (center_x, 0), (center_x, canvas_size), (200, 200, 200), 1)  # Y-axis
        cv2.line(canvas, (0, center_y), (canvas_size, center_y), (200, 200, 200), 1)  # X-axis
        
        # Draw board boundary circle
        cv2.circle(canvas, (center_x, center_y), int(self.board_radius * scale_factor), (0, 0, 0), 2)
        
        # Draw camera positions
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cv2.circle(canvas, cam1_px, 8, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        cv2.circle(canvas, cam2_px, 8, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw known segment markers for reference
        for segment, (x, y) in self.board_segments.items():
            segment_px = mm_to_canvas_px(x, y)
            cv2.circle(canvas, segment_px, 5, (128, 0, 128), -1)  # Purple dot
            cv2.putText(canvas, f"Seg {segment}", (segment_px[0]+5, segment_px[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
        
        # Draw vectors if available
        if self.cam1_vector is not None:
            board_point = mm_to_canvas_px(*self.cam1_vector)
            cv2.circle(canvas, board_point, 5, (0, 0, 255), -1)
            cv2.line(canvas, cam1_px, board_point, (0, 0, 255), 2)
            
            # Show angle information if available
            if self.cam1_angle is not None:
                angle_text = f"Cam1 Angle: {self.cam1_angle:.1f}°"
                cv2.putText(canvas, angle_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw angle representation on board
                if self.final_tip is not None:
                    dart_px = mm_to_canvas_px(*self.final_tip)
                    angle_rad = math.radians(90 - self.cam1_angle)  # Convert to radians from horizontal
                    dx = 30 * math.cos(angle_rad)
                    dy = 30 * math.sin(angle_rad)
                    end_x = int(dart_px[0] + dx)
                    end_y = int(dart_px[1] + dy)
                    cv2.line(canvas, dart_px, (end_x, end_y), (0, 0, 255), 2)
        
        if self.cam2_vector is not None:
            board_point = mm_to_canvas_px(*self.cam2_vector)
            cv2.circle(canvas, board_point, 5, (255, 0, 0), -1)
            cv2.line(canvas, cam2_px, board_point, (255, 0, 0), 2)
            
            # Show angle information if available
            if self.cam2_angle is not None:
                angle_text = f"Cam2 Angle: {self.cam2_angle:.1f}°"
                cv2.putText(canvas, angle_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw angle representation on board
                if self.final_tip is not None:
                    dart_px = mm_to_canvas_px(*self.final_tip)
                    angle_rad = math.radians(90 - self.cam2_angle)  # Convert to radians from horizontal
                    dx = 30 * math.cos(angle_rad)
                    dy = 30 * math.sin(angle_rad)
                    end_x = int(dart_px[0] + dx)
                    end_y = int(dart_px[1] + dy)
                    cv2.line(canvas, dart_px, (end_x, end_y), (255, 0, 0), 2)
        
        # Draw intersection point if available
        if self.final_tip is not None:
            # Calculate distance from center
            distance_mm = math.sqrt(self.final_tip[0]**2 + self.final_tip[1]**2)
            
            # Convert to canvas pixel coordinates
            dart_px = mm_to_canvas_px(*self.final_tip)
            
            # Draw the circle indicator
            cv2.circle(canvas, dart_px, 8, (0, 0, 0), -1)  # Black outline
            cv2.circle(canvas, dart_px, 6, (0, 255, 0), -1)  # Green center
            
            # Find closest segment
            closest_segment = None
            min_distance = float('inf')
            for segment, (seg_x, seg_y) in self.board_segments.items():
                dist = math.sqrt((self.final_tip[0] - seg_x)**2 + (self.final_tip[1] - seg_y)**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_segment = segment
            
            # Prepare the text info
            segment_info = f" (near Seg {closest_segment})" if closest_segment and min_distance < 50 else ""
            position_text = f"Dart: ({self.final_tip[0]:.1f}, {self.final_tip[1]:.1f}){segment_info}"
            distance_text = f"Distance from center: {distance_mm:.1f}mm"
            
            # Draw text with black outline for visibility
            cv2.putText(canvas, position_text, (dart_px[0]+10, dart_px[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Outline
            cv2.putText(canvas, position_text, (dart_px[0]+10, dart_px[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Text
            
            cv2.putText(canvas, distance_text, (dart_px[0]+10, dart_px[1]+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Outline
            cv2.putText(canvas, distance_text, (dart_px[0]+10, dart_px[1]+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Text
        
        # Add detection status indicator
        if self.is_detecting:
            status_text = "DETECTION ACTIVE"
            cv2.putText(canvas, status_text, (center_x - 100, canvas_size - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return canvas
    
    def check_for_dart_detection(self, cam1_mask, cam2_mask):
        """
        Check if a dart has been detected in both camera feeds
        
        Args:
            cam1_mask: Binary mask from camera 1
            cam2_mask: Binary mask from camera 2
            
        Returns:
            bool: True if dart is detected, False otherwise
        """
        # Check for significant white pixels in both masks
        white_pixels_cam1 = np.sum(cam1_mask > 0)
        white_pixels_cam2 = np.sum(cam2_mask > 0)
        
        # Threshold for detection (adjust based on testing)
        threshold_cam1 = 100
        threshold_cam2 = 100
        
        # Detect if significant change in both cameras
        is_detected = (white_pixels_cam1 > threshold_cam1 and 
                      white_pixels_cam2 > threshold_cam2)
        
        # Update detection state
        current_time = time.time()
        
        if is_detected:
            # Start a new detection if not already detecting
            if not self.is_detecting:
                self.is_detecting = True
                print("Dart detected!")
            
            # Update the last detection time
            self.last_detection_time = current_time
        elif self.is_detecting:
            # Check if detection timeout has elapsed
            if current_time - self.last_detection_time > self.detection_timeout:
                self.is_detecting = False
                print("Detection completed.")
                
                # Reset buffers
                self._initialize_buffers()
        
        return self.is_detecting
    
    def save_config(self, filename="dart_detector_config.json"):
        """
        Save the current configuration to a JSON file
        
        Args:
            filename: Path to save the configuration
        """
        import json
        
        # Convert some values to be JSON serializable
        json_config = self.config.copy()
        
        # Convert tuples to lists
        for key, value in json_config.items():
            if isinstance(value, tuple):
                json_config[key] = list(value)
            elif isinstance(value, dict):
                new_dict = {}
                for k, v in value.items():
                    if isinstance(k, tuple):
                        str_k = str(k)
                        new_dict[str_k] = v if not isinstance(v, tuple) else list(v)
                    else:
                        new_dict[k] = v if not isinstance(v, tuple) else list(v)
                json_config[key] = new_dict
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(json_config, f, indent=4)
        
        print(f"Configuration saved to {filename}")
    
    def load_config(self, filename="dart_detector_config.json"):
        """
        Load configuration from a JSON file
        
        Args:
            filename: Path to the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        import json
        
        try:
            with open(filename, 'r') as f:
                loaded_config = json.load(f)
            
            # Convert lists back to tuples and handle special cases
            for key, value in loaded_config.items():
                if isinstance(value, list):
                    loaded_config[key] = tuple(value)
                elif isinstance(value, dict):
                    # Special handling for dictionaries with tuple keys
                    if key == "calibration_points":
                        calibration_dict = {}
                        for k_str, v in value.items():
                            # Handle string representations of tuples like "(0, 0)"
                            if k_str.startswith("(") and k_str.endswith(")"):
                                try:
                                    # Parse the string back to a tuple
                                    parts = k_str.strip("()").split(",")
                                    tuple_key = (float(parts[0].strip()), float(parts[1].strip()))
                                    calibration_dict[tuple_key] = tuple(v) if isinstance(v, list) else v
                                except:
                                    # If parsing fails, use the string as is
                                    calibration_dict[k_str] = tuple(v) if isinstance(v, list) else v
                            else:
                                # Regular keys
                                calibration_dict[k_str] = tuple(v) if isinstance(v, list) else v
                        loaded_config[key] = calibration_dict
                    else:
                        # Handle regular dictionaries
                        new_dict = {}
                        for k, v in value.items():
                            new_dict[int(k) if k.isdigit() else k] = tuple(v) if isinstance(v, list) else v
                        loaded_config[key] = new_dict
            
            # Update config
            self.config.update(loaded_config)
            
            # Reinitialize with new settings
            self._initialize_settings()
            self._initialize_background_subtractors()
            self._initialize_buffers()
            
            print(f"Configuration loaded from {filename}")
            return True
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def calibrate_interactively(self):
        """
        Interactive calibration procedure to fine-tune the system
        This method allows adjusting parameters in real-time
        """
        print("Starting interactive calibration mode...")
        print("Press 'q' to exit calibration.")
        print("Press 's' to save current configuration.")
        print("Press 'b' to capture background.")
        
        # Open cameras
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both cameras.")
            return
        
        # Create windows with trackbars
        cv2.namedWindow("Calibration Controls")
        
        # Add trackbars for ROI settings
        cv2.createTrackbar("Cam1 Board Plane Y", "Calibration Controls", self.cam1_board_plane_y, 
                           self.frame_height, lambda x: self._update_param("cam1_board_plane_y", x))
        cv2.createTrackbar("Cam1 ROI Range", "Calibration Controls", self.cam1_roi_range, 
                           100, lambda x: self._update_param("cam1_roi_range", x))
        cv2.createTrackbar("Cam2 Board Plane Y", "Calibration Controls", self.cam2_board_plane_y, 
                           self.frame_height, lambda x: self._update_param("cam2_board_plane_y", x))
        cv2.createTrackbar("Cam2 ROI Range", "Calibration Controls", self.cam2_roi_range, 
                           100, lambda x: self._update_param("cam2_roi_range", x))
        
        # Add trackbars for background subtraction
        cv2.createTrackbar("History", "Calibration Controls", self.config["history"], 
                           300, lambda x: self._update_param("history", x))
        cv2.createTrackbar("Var Threshold", "Calibration Controls", self.config["var_threshold"], 
                           100, lambda x: self._update_param("var_threshold", x))
        cv2.createTrackbar("Learning Rate *100", "Calibration Controls", int(self.config["learning_rate"]*100), 
                           100, lambda x: self._update_param("learning_rate", x/100.0))
        
        # Add trackbars for contour filtering
        cv2.createTrackbar("Min Contour Area", "Calibration Controls", self.config["min_contour_area"], 
                           100, lambda x: self._update_param("min_contour_area", x))
        
        # Add trackbars for angle detection
        cv2.createTrackbar("Angle Search Depth", "Calibration Controls", self.config["angle_search_depth"], 
                           50, lambda x: self._update_param("angle_search_depth", x))
        cv2.createTrackbar("Angle Search Width", "Calibration Controls", self.config["angle_search_width"], 
                           100, lambda x: self._update_param("angle_search_width", x))
        
        try:
            while True:
                # Read frames
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print("Error: Failed to grab frames.")
                    break
                
                # Process frames
                proc_frame1, mask1 = self.process_camera1_frame(frame1)
                proc_frame2, mask2 = self.process_camera2_frame(frame2)
                
                # Check for dart detection
                self.check_for_dart_detection(mask1, mask2)
                
                # Calculate intersection of camera vectors
                if self.cam1_vector is not None and self.cam2_vector is not None:
                    new_tip = self.compute_intersection()
                    if new_tip is not None:
                        self.final_tip = self.apply_temporal_filtering(new_tip, self.final_tip_buffer)
                else:
                    self.final_tip = None
                
                # Create board visualization
                board_vis = self.create_board_visualization()
                
                # Create control panel display
                control_panel = np.ones((400, 400, 3), dtype=np.uint8) * 255
                
                # Add current configuration values to control panel
                param_texts = [
                    f"Cam1 Board Y: {self.cam1_board_plane_y}",
                    f"Cam1 ROI Range: {self.cam1_roi_range}",
                    f"Cam2 Board Y: {self.cam2_board_plane_y}",
                    f"Cam2 ROI Range: {self.cam2_roi_range}",
                    f"BG History: {self.config['history']}",
                    f"BG Threshold: {self.config['var_threshold']}",
                    f"Learning Rate: {self.config['learning_rate']:.3f}",
                    f"Min Contour: {self.config['min_contour_area']}",
                    f"Angle Depth: {self.config['angle_search_depth']}",
                    f"Angle Width: {self.config['angle_search_width']}",
                ]
                
                for i, text in enumerate(param_texts):
                    cv2.putText(control_panel, text, (10, 20 + i*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Display help text
                cv2.putText(control_panel, "Press 'q' to exit", (10, 360), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(control_panel, "Press 's' to save config", (10, 380), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # Display results
                cv2.imshow("Camera 1", proc_frame1)
                cv2.imshow("Camera 1 Mask", mask1)
                cv2.imshow("Camera 2", proc_frame2)
                cv2.imshow("Camera 2 Mask", mask2)
                cv2.imshow("Board Visualization", board_vis)
                cv2.imshow("Calibration Controls", control_panel)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_config()
                elif key == ord('b'):
                    print("Capturing background...")
                    # Reset background subtractors
                    self._initialize_background_subtractors()
                    
                    # Allow time for background to be captured
                    for _ in range(30):
                        ret1, frame1 = cap1.read()
                        ret2, frame2 = cap2.read()
                        if ret1 and ret2:
                            frame1_rot = cv2.rotate(frame1, cv2.ROTATE_180)
                            frame2_rot = cv2.rotate(frame2, cv2.ROTATE_180)
                            
                            roi1 = frame1_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
                            roi2 = frame2_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
                            
                            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
                            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
                            
                            # Apply with high learning rate to capture background quickly
                            self.bg_subtractor1.apply(gray1, learningRate=0.1)
                            self.bg_subtractor2.apply(gray2, learningRate=0.1)
                            
                            cv2.waitKey(1)
                    
                    print("Background captured.")
        
        finally:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            print("Calibration completed.")
    
    def _update_param(self, param_name, value):
        """
        Update a parameter in the configuration
        
        Args:
            param_name: Name of the parameter
            value: New value
        """
        # Update configuration
        self.config[param_name] = value
        
        # Update relevant settings
        if param_name == "cam1_board_plane_y":
            self.cam1_board_plane_y = value
            self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
            self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        elif param_name == "cam1_roi_range":
            self.cam1_roi_range = value
            self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
            self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        elif param_name == "cam2_board_plane_y":
            self.cam2_board_plane_y = value
            self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
            self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        elif param_name == "cam2_roi_range":
            self.cam2_roi_range = value
            self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
            self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        elif param_name in ["history", "var_threshold"]:
            # Reinitialize background subtractors
            self._initialize_background_subtractors()
    
    def run(self):
        """Main processing loop"""
        # Open cameras
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both cameras.")
            return
        
        print("Camera opened successfully. Press 'q' to exit.")
        
        try:
            while self.running:
                # Read frames
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print("Error: Failed to grab frames.")
                    break
                
                # Process frames
                proc_frame1, mask1 = self.process_camera1_frame(frame1)
                proc_frame2, mask2 = self.process_camera2_frame(frame2)
                
                # Check for dart detection
                self.check_for_dart_detection(mask1, mask2)
                
                # Calculate intersection of camera vectors
                if self.cam1_vector is not None and self.cam2_vector is not None:
                    new_tip = self.compute_intersection()
                    if new_tip is not None:
                        self.final_tip = self.apply_temporal_filtering(new_tip, self.final_tip_buffer)
                else:
                    self.final_tip = None
                
                # Create board visualization
                board_vis = self.create_board_visualization()
                
                # Display results
                cv2.imshow("Camera 1", proc_frame1)
                cv2.imshow("Camera 1 Mask", mask1)
                cv2.imshow("Camera 2", proc_frame2)
                cv2.imshow("Camera 2 Mask", mask2)
                cv2.imshow("Board Visualization", board_vis)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            print("Cameras released.")

# Utility Functions for Configuration and Testing

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

def optimize_noise_reduction():
    """
    Optimize background subtraction and noise reduction parameters
    for maximum stability and sensitivity
    """
    print("Starting Noise Reduction Optimization")
    print("====================================")
    print("This will automatically test different parameter combinations")
    print("to find the optimal noise reduction settings.")
    print("\nThe process requires a dart to be placed and removed several times.")
    print("Follow the prompts during the optimization process.")
    
    # Create detector with current configuration
    detector = EnhancedDartDetector()
    detector.load_config()
    
    # Parameters to optimize
    param_ranges = {
        "history": [50, 75, 100, 125, 150, 200],
        "var_threshold": [15, 20, 25, 30, 35, 40, 45],
        "learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05],
        "min_contour_area": [5, 10, 15, 20, 25, 30]
    }
    
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
    
    # Set up the test cases
    test_cases = []
    
    # Generate test combinations (limited subset to avoid too many tests)
    # Using Latin Hypercube Sampling for more efficient parameter space exploration
    num_tests = 15
    for _ in range(num_tests):
        test_case = {}
        for param, values in param_ranges.items():
            # Select a random value from the range
            test_case[param] = random.choice(values)
        test_cases.append(test_case)
    
    # Keep the current settings as one test case
    current_settings = {
        "history": detector.config["history"],
        "var_threshold": detector.config["var_threshold"],
        "learning_rate": detector.config["learning_rate"],
        "min_contour_area": detector.config["min_contour_area"]
    }
    test_cases.append(current_settings)
    
    # Test each combination
    results = []
    for test_idx, test_case in enumerate(test_cases):
        print(f"\nTest {test_idx+1}/{len(test_cases)} - Parameters:")
        for param, value in test_case.items():
            print(f"  {param}: {value}")
            detector.config[param] = value
        
        # Reinitialize for this test
        detector._initialize_background_subtractors()
        
        # Initialize background
        print("\nCapturing background (ensure no dart is visible)...")
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
        
        # Ask user to place dart
        print("\nPlace a dart on the board.")
        print("Press Enter when ready.")
        input()
        
        # Measure detection quality with dart present
        dart_present_detections = 0
        false_negatives = 0
        
        for _ in range(20):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                continue
            
            proc_frame1, mask1 = detector.process_camera1_frame(frame1)
            proc_frame2, mask2 = detector.process_camera2_frame(frame2)
            
            # Check detection
            is_detected = detector.check_for_dart_detection(mask1, mask2)
            
            if is_detected:
                dart_present_detections += 1
            else:
                false_negatives += 1
            
            # Display visualization
            board_vis = detector.create_board_visualization()
            cv2.imshow("Test Visualization", board_vis)
            cv2.waitKey(100)
        
        # Ask user to remove dart
        print("\nRemove the dart from the board.")
        print("Press Enter when ready.")
        input()
        
        # Measure false positives with no dart present
        false_positives = 0
        
        for _ in range(20):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                continue
            
            proc_frame1, mask1 = detector.process_camera1_frame(frame1)
            proc_frame2, mask2 = detector.process_camera2_frame(frame2)
            
            # Check detection
            is_detected = detector.check_for_dart_detection(mask1, mask2)
            
            if is_detected:
                false_positives += 1
            
            # Display visualization
            board_vis = detector.create_board_visualization()
            cv2.imshow("Test Visualization", board_vis)
            cv2.waitKey(100)
        
        # Calculate score for this test case
        detection_rate = dart_present_detections / 20.0
        false_negative_rate = false_negatives / 20.0
        false_positive_rate = false_positives / 20.0
        
        # Weighted score (prioritize detection over false positives)
        score = detection_rate * 0.6 - false_positive_rate * 0.4
        
        # Save result
        result = {
            "parameters": test_case,
            "detection_rate": detection_rate,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "score": score
        }
        results.append(result)
        
        print(f"\nTest results:")
        print(f"  Detection rate: {detection_rate * 100:.1f}%")
        print(f"  False negative rate: {false_negative_rate * 100:.1f}%")
        print(f"  False positive rate: {false_positive_rate * 100:.1f}%")
        print(f"  Score: {score:.3f}")
    
    # Find best result
    best_result = max(results, key=lambda x: x["score"])
    
    print("\n\nOptimization completed!")
    print("\nBest parameters:")
    for param, value in best_result["parameters"].items():
        print(f"  {param}: {value}")
    
    print(f"\nBest score: {best_result['score']:.3f}")
    print(f"Detection rate: {best_result['detection_rate'] * 100:.1f}%")
    print(f"False negative rate: {best_result['false_negative_rate'] * 100:.1f}%")
    print(f"False positive rate: {best_result['false_positive_rate'] * 100:.1f}%")
    
    # Ask to apply the best settings
    print("\nDo you want to apply these optimized settings? (y/n)")
    apply_settings = input().lower()
    
    if apply_settings == 'y':
        # Update configuration with best parameters
        for param, value in best_result["parameters"].items():
            detector.config[param] = value
        
        # Save configuration
        detector.save_config()
        print("Optimized settings applied and saved.")
    
    # Clean up
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def optimize_angle_detection():
    """
    Optimize parameters related to angle detection accuracy
    """
    print("Starting Angle Detection Optimization")
    print("===================================")
    print("This will test different angle detection parameters")
    print("to find the optimal settings for accurate angle measurement.")
    
    # Create detector with current configuration
    detector = EnhancedDartDetector()
    detector.load_config()
    
    # Parameters to optimize
    param_ranges = {
        "angle_search_depth": [15, 20, 25, 30, 35, 40],
        "angle_search_width": [20, 30, 40, 50, 60],
        "min_points_for_angle": [5, 8, 10, 12, 15],
        "angle_buffer_size": [3, 5, 7, 9, 11]
    }
    
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
    
    # Initialize background
    print("\nCapturing background (ensure no dart is visible)...")
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
    
    # Set up the test cases
    test_cases = []
    
    # Generate test combinations (limited subset)
    num_tests = 10
    for _ in range(num_tests):
        test_case = {}
        for param, values in param_ranges.items():
            test_case[param] = random.choice(values)
        test_cases.append(test_case)
    
    # Keep the current settings as one test case
    current_settings = {
        "angle_search_depth": detector.config["angle_search_depth"],
        "angle_search_width": detector.config["angle_search_width"],
        "min_points_for_angle": detector.config["min_points_for_angle"],
        "angle_buffer_size": detector.config["angle_buffer_size"]
    }
    test_cases.append(current_settings)
    
    # Angles to test
    test_angles = ["VERTICAL", "LEFT", "RIGHT"]
    
    # Test each combination
    results = []
    for test_idx, test_case in enumerate(test_cases):
        print(f"\nTest {test_idx+1}/{len(test_cases)} - Parameters:")
        for param, value in test_case.items():
            print(f"  {param}: {value}")
            detector.config[param] = value
        
        # Reinitialize buffers
        detector._initialize_buffers()
        
        angle_stabilities = []
        
        # Test with different dart angles
        for angle_type in test_angles:
            print(f"\nTest with dart at {angle_type} angle.")
            print("Place a dart on the board with the tip pointing {}.".format(
                "vertically" if angle_type == "VERTICAL" else 
                "leaning to the left" if angle_type == "LEFT" else "leaning to the right"
            ))
            print("Press Enter when ready.")
            input()
            
            # Measure angle stability
            angle_measurements1 = []
            angle_measurements2 = []
            
            for _ in range(20):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    continue
                
                proc_frame1, mask1 = detector.process_camera1_frame(frame1)
                proc_frame2, mask2 = detector.process_camera2_frame(frame2)
                
                if detector.cam1_angle is not None:
                    angle_measurements1.append(detector.cam1_angle)
                
                if detector.cam2_angle is not None:
                    angle_measurements2.append(detector.cam2_angle)
                
                # Display visualization
                board_vis = detector.create_board_visualization()
                cv2.imshow("Test Visualization", board_vis)
                cv2.waitKey(100)
            
            # Calculate angle stability (standard deviation)
            if len(angle_measurements1) > 0:
                stability1 = np.std(angle_measurements1)
            else:
                stability1 = float('inf')
                
            if len(angle_measurements2) > 0:
                stability2 = np.std(angle_measurements2)
            else:
                stability2 = float('inf')
                
            # Average stability (lower is better)
            avg_stability = (stability1 + stability2) / 2.0
            
            angle_stabilities.append(avg_stability)
            
            print(f"Angle stability for {angle_type}: {avg_stability:.2f}° std dev")
        
        # Calculate overall score for this test case
        if len(angle_stabilities) > 0:
            avg_angle_stability = np.mean(angle_stabilities)
            score = 10.0 / (1.0 + avg_angle_stability)  # Higher score for lower variability
        else:
            avg_angle_stability = float('inf')
            score = 0.0
        
        # Save result
        result = {
            "parameters": test_case,
            "avg_angle_stability": avg_angle_stability,
            "individual_stabilities": angle_stabilities,
            "score": score
        }
        results.append(result)
        
        print(f"\nTest results:")
        print(f"  Average angle stability: {avg_angle_stability:.2f}° std dev")
        print(f"  Score: {score:.3f}")
    
    # Find best result
    best_result = max(results, key=lambda x: x["score"])
    
    print("\n\nOptimization completed!")
    print("\nBest parameters:")
    for param, value in best_result["parameters"].items():
        print(f"  {param}: {value}")
    
    print(f"\nBest score: {best_result['score']:.3f}")
    print(f"Average angle stability: {best_result['avg_angle_stability']:.2f}° std dev")
    
    # Ask to apply the best settings
    print("\nDo you want to apply these optimized settings? (y/n)")
    apply_settings = input().lower()
    
    if apply_settings == 'y':
        # Update configuration with best parameters
        for param, value in best_result["parameters"].items():
            detector.config[param] = value
        
        # Save configuration
        detector.save_config()
        print("Optimized settings applied and saved.")
    
    # Clean up
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def reset_configuration():
    """Reset to default configuration"""
    print("Resetting to default configuration...")
    
    # Create a fresh detector instance with default settings
    detector = EnhancedDartDetector()
    
    # Save the default configuration
    detector.save_config()
    
    print("Configuration has been reset to defaults.")
    print("Default settings saved to dart_detector_config.json")

def run_auto_optimization():
    """Run full auto-optimization process"""
    print("Starting Full Auto-Optimization")
    print("==============================")
    print("This will run a complete optimization process including:")
    print("1. Noise reduction optimization")
    print("2. Angle detection optimization")
    print("3. Accuracy testing")
    print("\nThis process will take some time and require interaction.")
    print("Continue? (y/n)")
    
    choice = input().lower()
    if choice != 'y':
        print("Optimization cancelled.")
        return
    
    # Run optimizations
    optimize_noise_reduction()
    optimize_angle_detection()
    test_accuracy()
    
    print("\nFull optimization process completed!")
    print("Configuration has been saved to dart_detector_config.json")
    print("Accuracy test results saved to accuracy_test_results.json")

# Main Program

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Dual Camera Dart Detector")
    parser.add_argument("--calibrate", action="store_true", help="Run interactive calibration")
    parser.add_argument("--test", action="store_true", help="Run accuracy test")
    parser.add_argument("--optimize", action="store_true", help="Run auto-optimization")
    parser.add_argument("--reset", action="store_true", help="Reset to default configuration")
    parser.add_argument("--noise", action="store_true", help="Optimize noise reduction")
    parser.add_argument("--angle", action="store_true", help="Optimize angle detection")
    parser.add_argument("--cam1", type=int, default=0, help="Camera 1 index (default: 0)")
    parser.add_argument("--cam2", type=int, default=2, help="Camera 2 index (default: 2)")
    
    args = parser.parse_args()
    
    # Run the selected mode
    if args.calibrate:
        run_interactive_calibration()
    elif args.test:
        test_accuracy()
    elif args.optimize:
        run_auto_optimization()
    elif args.reset:
        reset_configuration()
    elif args.noise:
        optimize_noise_reduction()
    elif args.angle:
        optimize_angle_detection()
    else:
        # Run normal detection mode
        detector = EnhancedDartDetector(cam_index1=args.cam1, cam_index2=args.cam2)
        detector.load_config()
        detector.run()
