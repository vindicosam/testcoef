import cv2
import numpy as np
import math
import time
from collections import deque
from bisect import bisect_left
import json
import ast

def debug_image_paths():
    import os
    print(f"Current working directory: {os.getcwd()}")
    image_name = "winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg"
    print(f"Looking for image: {image_name}")
    print(f"File exists: {os.path.exists(image_name)}")
    return os.path.join(os.getcwd(), image_name)

class OptimizedDartTracker:
    def __init__(self, cam_index1=0, cam_index2=2):
        """
        Optimized dart tracking system that uses existing calibration points
        and enhanced detection algorithms to achieve 1mm accuracy
        """
        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2
        
        # Camera settings
        self.frame_width = 640
        self.frame_height = 480

        # Static camera positions
        self.camera1_position = (0, 550)    # Front camera fixed position
        self.camera2_position = (-400, 0)   # Side camera fixed position

        # Board ROI settings from original code
        self.cam1_board_plane_y = 182
        self.cam1_roi_range = 30
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        self.cam2_board_plane_y = 193
        self.cam2_roi_range = 30
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        
        # Initialize with original calibration points
        self.calibration_points = {
            # Format: (board_x, board_y): (cam1_pixel_x, cam2_pixel_x)
            (0, 0): (306, 316),
            (-171, 0): (580, 384),
            (171, 0): (32, 294),
            (0, 171): (304, 28),
            (0, -171): (305, 572),
            (90, 50): (151, 249),
            (-20, 103): (327, 131),
            (20, -100): (277, 459),
            (90, -50): (359, 406),
            (114, 121, 17, 153): (17, 153),
            (48, 86, 214, 182): (214, 182),
            (119, -117, 167, 429): (167, 429),
            (86, -48, 189, 359): (189, 359),
            (-118, -121, 453, 624): (453, 624),
            (-50, -88, 373, 478): (373, 478),
            (-121, 118, 624, 240): (624, 240),
            (-90, 47, 483, 42): (483, 42)
        }
        
        # Convert any tuples with 4 elements to proper format
        calibration_points_fixed = {}
        for k, v in self.calibration_points.items():
            if len(k) == 4:  # It's in the format (x, y, px1, px2)
                calibration_points_fixed[(k[0], k[1])] = (k[2], k[3])
            else:
                calibration_points_fixed[k] = v
        
        self.calibration_points = calibration_points_fixed
        
        # Import segment calibration points from the original code
        segment_calibration_points = [
            # Double segments (outer ring)
            (0, 169, 394, 31),      # Double 20 (top)
            (52, 161, 145, 80),     # Double 1
            (98, 139, 33, 133),     # Double 18
            (139, 98, 7, 189),     # Double 4
            (161, 52, 18, 241),     # Double 13
            (169, 0, 51, 296),      # Double 6 (right)
            (161, -52, 97, 349),    # Double 10
            (139, -98, 153, 405),   # Double 15
            (98, -139, 208, 462),   # Double 2
            (52, -161, 263, 517),   # Double 17
            (0, -169, 317, 567),    # Double 3 (bottom)
            (-52, -161, 371, 608),  # Double 19
            (-98, -139, 429, 629),  # Double 7
            (-139, -98, 490, 608),  # Double 16
            (-161, -52, 545, 518),  # Double 8
            (-169, 0, 592, 357),    # Double 11 (left)
            (-161, 52, 629, 209),   # Double 14
            (-139, 98, 636, 82),    # Double 9
            (-98, 139, 597, 17),    # Double 12
            (-52, 161, 486, 9),     # Double 5

            # Triple segments (middle ring)
            (0, 106, 321, 145),     # Triple 20 (top)
            (33, 101, 249, 164),    # Triple 1
            (62, 87, 191, 192),     # Triple 18
            (87, 62, 155, 227),     # Triple 4
            (101, 33, 143, 265),    # Triple 13
            (106, 0, 155, 304),     # Triple 6 (right)
            (101, -33, 175, 345),   # Triple 10
            (87, -62, 209, 382),    # Triple 15
            (62, -87, 244, 419),    # Triple 2
            (33, -101, 281, 452),   # Triple 17
            (0, -106, 321, 478),    # Triple 3 (bottom)
            (-33, -101, 360, 494),  # Triple 19
            (-62, -87, 398, 490),   # Triple 7
            (-87, -62, 436, 466),   # Triple 16
            (-101, -33, 466, 414),  # Triple 8
            (-106, 0, 489, 345),    # Triple 11 (left)
            (-101, 33, 497, 274),   # Triple 14
            (-87, 62, 490, 207),    # Triple 9
            (-62, 87, 456, 162),    # Triple 12
            (-33, 101, 395, 143),   # Triple 5
            (0, 171, 318, 35),
            (0, 161, 318, 58),
            (0, 147, 318, 84),
            (0, 136, 318, 105),
            (0, 122, 318, 124),
            (0, 106, 318, 144),
            (0, 98, 318, 167),
            (0, 89, 318, 179),
            (0, 82, 318, 192),
            (0, 74, 318, 205),
            (0, 66, 318, 215),
            (0, 61, 318, 226),
            (0, 54, 318, 237),
            (0, 44, 318, 249),
            (0, 37, 318, 265),
            (0, 29, 318, 276),
            (0, 20, 318, 292),
            (0, 5.85, 318, 313),
            (0, 6.85, 318, 308),
            (0, 14.9, 318, 298),
            (0, -5.85, 318, 328),
            (0, -6.85, 318, 334),
            (0, -14.9, 318, 344),
            (0, -16, 318, 349),
            (0, -22, 318, 359),
            (0, -29, 318, 369),
            (0, -36, 318, 379),
            (0, -43, 318, 390),
            (0, -51, 318, 400),
            (0, -60, 318, 414),
            (0, -68, 318, 426),
            (0, -77, 318, 438),
            (0, -84, 318, 447),
            (0, -96, 318, 465),
            (0, -108, 318, 485),
            (0, -116, 318, 498),
            (0, -124, 318, 510),
            (0, -132, 318, 519),
            (0, -140, 318, 530),
            (0, -150, 318, 545),
            (0, -162, 318, 558),
        ]
        
        # Add segment points to calibration
        for point in segment_calibration_points:
            board_x, board_y, cam1_pixel_x, cam2_pixel_x = point
            if cam1_pixel_x is not None and cam2_pixel_x is not None:
                self.calibration_points[(board_x, board_y)] = (cam1_pixel_x, cam2_pixel_x)
        
        # Create mapping tables for interpolation
        self.cam1_pixel_to_board_mapping = []
        self.cam2_pixel_to_board_mapping = []
        
        # Fill the mapping tables
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            # We only need x mapping for camera 1
            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
            # We only need y mapping for camera 2
            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
        
        # Sort the mappings by pixel values for efficient lookup
        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
        
        # Background subtractors with optimized parameters
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
            history=3000,        # Longer history for more stable background
            varThreshold=60,     # Slightly reduced threshold for better detection
            detectShadows=False
        )
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
            history=3000,
            varThreshold=60,
            detectShadows=False
        )
        
        # Board information
        self.board_extent = 171
        self.board_radius = 170
        
        # Detection vectors
        self.cam1_vector = None
        self.cam2_vector = None
        self.final_tip = None
        
        # Enhanced detection history for better smoothing
        self.detection_history = {
            'cam1': deque(maxlen=5),  # Increased history size
            'cam2': deque(maxlen=5),
            'final': deque(maxlen=5)
        }
        
        # Kalman filter for smooth tracking
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kalman.processNoiseCov = np.array([
            [1e-4, 0, 0, 0],
            [0, 1e-4, 0, 0],
            [0, 0, 1e-2, 0],
            [0, 0, 0, 1e-2]
        ], np.float32)
        self.kalman.measurementNoiseCov = np.array([
            [1e-1, 0],
            [0, 1e-1]
        ], np.float32)
        self.kalman_initialized = False
        
        # Board segments for scoring
        self.board_segments = {
            20: (0, 169),
            1: (52, 161),
            18: (98, 139),
            4: (139, 98),
            13: (161, 52),
            6: (169, 0),
            10: (161, -52),
            15: (139, -98),
            2: (98, -139),
            17: (52, -161),
            3: (0, -169),
            19: (-52, -161),
            7: (-98, -139),
            16: (-139, -98),
            8: (-161, -52),
            11: (-169, 0),
            14: (-161, 52),
            9: (-139, 98),
            12: (-98, 139),
            5: (-52, 161)
        }
        
        # Load dartboard image
        self.board_image = cv2.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        if self.board_image is None:
            print("Warning: dartboard image not found. Using placeholder.")
            self.board_image = np.zeros((500, 500, 3), dtype=np.uint8)
            # Draw a simple dartboard representation
            cv2.circle(self.board_image, (250, 250), 250, (0, 0, 128), -1)
            cv2.circle(self.board_image, (250, 250), 200, (0, 128, 128), -1)
            cv2.circle(self.board_image, (250, 250), 170, (0, 0, 128), 3)  # Double ring
            cv2.circle(self.board_image, (250, 250), 107, (0, 0, 128), 3)  # Triple ring
            cv2.circle(self.board_image, (250, 250), 32, (0, 128, 128), -1)  # Outer bull
            cv2.circle(self.board_image, (250, 250), 13, (0, 0, 128), -1)  # Inner bull
        
        # State tracking
        self.calibration_mode = False
        self.calibration_point = None
        
        # Sub-pixel refinement settings
        self.subpixel_window = (5, 5)
        self.subpixel_iterations = 30
        self.subpixel_epsilon = 0.001
        
        # Filtering parameters for 1mm accuracy
        self.movement_threshold = 1.0  # mm - threshold for stable detection
        self.stable_position_count = 0  # counter for stable positions
        self.last_stable_position = None  # last stable position
        
        # Advanced interpolation
        self.use_advanced_interpolation = True  # Enable advanced interpolation
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
    
    def load_calibration(self, filename="dart_calibration.json"):
        """Load calibration from file"""
        try:
            import json
            import ast
            with open(filename, "r") as f:
                loaded_points = json.load(f)
                self.calibration_points = {ast.literal_eval(k): v for k, v in loaded_points.items()}
            
            # Rebuild mapping tables
            self.cam1_pixel_to_board_mapping = []
            self.cam2_pixel_to_board_mapping = []
            for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
                self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
                self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
            self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
            self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
            
            print(f"Loaded {len(self.calibration_points)} calibration points")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def save_calibration(self, filename="dart_calibration.json"):
        """Save calibration to file"""
        try:
            with open(filename, "w") as f:
                json.dump({str(k): v for k, v in self.calibration_points.items()}, f)
            print(f"Saved {len(self.calibration_points)} calibration points")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def interpolate_value(self, pixel_value, mapping_table):
        """
        Interpolate a value using the provided mapping table.
        mapping_table is a list of (pixel_value, board_coordinate) pairs sorted by pixel_value.
        """
        # Handle edge cases
        if not mapping_table:
            return None
        
        # If pixel value is outside the range of our mapping, use the nearest edge value
        if pixel_value <= mapping_table[0][0]:
            return mapping_table[0][1]
        if pixel_value >= mapping_table[-1][0]:
            return mapping_table[-1][1]
        
        # Find position where pixel_value would be inserted to maintain sorted order
        pos = bisect_left([x[0] for x in mapping_table], pixel_value)
        
        # If exact match
        if pos < len(mapping_table) and mapping_table[pos][0] == pixel_value:
            return mapping_table[pos][1]
        
        # Need to interpolate between pos-1 and pos
        lower_pixel, lower_value = mapping_table[pos-1]
        upper_pixel, upper_value = mapping_table[pos]
        
        # Linear interpolation
        ratio = (pixel_value - lower_pixel) / (upper_pixel - lower_pixel)
        interpolated_value = lower_value + ratio * (upper_value - lower_value)
        
        return interpolated_value
    
    def advanced_interpolation(self, cam1_pixel_x, cam2_pixel_x):
        """
        Advanced interpolation using nearby calibration points for better accuracy
        """
        # Find nearby calibration points in pixel space
        nearby_points = []
        for (board_x, board_y), (cal_cam1_px, cal_cam2_px) in self.calibration_points.items():
            # Calculate weighted distance in pixel space
            dx = (cam1_pixel_x - cal_cam1_px)
            dy = (cam2_pixel_x - cal_cam2_px)
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Add points within reasonable distance
            if distance < 100:  # Within 100 pixels
                nearby_points.append(((board_x, board_y), distance))
        
        # If we have enough nearby points
        if len(nearby_points) >= 3:
            # Sort by distance
            nearby_points.sort(key=lambda x: x[1])
            
            # Use inverse distance weighting with power parameter for interpolation
            power = 2  # IDW power parameter
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for (board_x, board_y), distance in nearby_points[:5]:  # Use closest 5 points
                # Add small constant to avoid division by zero
                weight = 1.0 / (distance + 0.1) ** power
                total_weight += weight
                
                weighted_x += board_x * weight
                weighted_y += board_y * weight
            
            # Normalize
            if total_weight > 0:
                return weighted_x / total_weight, weighted_y / total_weight
        
        # Fallback to standard interpolation if advanced method fails
        return (
            self.interpolate_value(cam1_pixel_x, self.cam1_pixel_to_board_mapping),
            self.interpolate_value(cam2_pixel_x, self.cam2_pixel_to_board_mapping)
        )
    
    def apply_smoothing(self, new_value, history_key):
        """
        Apply smoothing to the detection using weighted moving average
        """
        if new_value is None:
            return None
        
        # Add new value to history
        self.detection_history[history_key].append(new_value)
        
        # If we have enough history, compute weighted average
        if len(self.detection_history[history_key]) >= 2:
            # More recent values have higher weight
            total_weight = 0
            weighted_sum = 0
            
            # For final position (x,y tuple)
            if history_key == 'final':
                weighted_x = 0
                weighted_y = 0
                
                # Calculate weights based on recency
                weights = [i+1 for i in range(len(self.detection_history[history_key]))]
                total_weight = sum(weights)
                
                for i, (x, y) in enumerate(self.detection_history[history_key]):
                    weight = weights[i]
                    weighted_x += x * weight
                    weighted_y += y * weight
                
                return (weighted_x / total_weight, weighted_y / total_weight)
            else:
                # For single values (cam1/cam2)
                weights = [i+1 for i in range(len(self.detection_history[history_key]))]
                total_weight = sum(weights)
                
                for i, value in enumerate(self.detection_history[history_key]):
                    weight = weights[i]
                    weighted_sum += value * weight
                
                return weighted_sum / total_weight
        
        return new_value
    
    def update_kalman(self, position):
        """Update Kalman filter with new position measurement"""
        if position is None:
            return
        
        measurement = np.array([[position[0]], [position[1]]], np.float32)
        
        if not self.kalman_initialized:
            # Initialize Kalman filter with first measurement
            self.kalman.statePre = np.array([[position[0]], [position[1]], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[position[0]], [position[1]], [0], [0]], np.float32)
            self.kalman_initialized = True
            return position
        
        # Prediction step
        prediction = self.kalman.predict()
        
        # Correction step
        corrected = self.kalman.correct(measurement)
        
        # Return corrected state (x, y)
        return (corrected[0, 0], corrected[1, 0])
    
    def detect_point_with_subpixel(self, contour, roi_center_y):
        """
        Detect dart with subpixel accuracy using moments
        
        Args:
            contour: Detected contour
            roi_center_y: Y-coordinate of ROI center line
            
        Returns:
            x: X-coordinate with subpixel accuracy
        """
        if cv2.contourArea(contour) < 5:
            return None
        
        # Get basic bounding box
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w / 2
        
        # Use moments for more accurate center (this already gives sub-pixel accuracy)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            center_x = M["m10"] / M["m00"]
        
        return center_x
    
    def process_camera1_frame(self, frame):
        """Process camera 1 frame with enhanced detection"""
        # Rotate frame 180 degrees as in original code
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor1.apply(blurred)
        
        # Threshold and clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to improve mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        dart_pixel_x = None
        best_contour = None
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        # Process contours
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                # Use enhanced subpixel detection
                dart_pixel_x = self.detect_point_with_subpixel(contour, roi_center_y)
                if dart_pixel_x is not None:
                    best_contour = contour
                    break
        
        # If dart detected
        if dart_pixel_x is not None:
            # Highlight the detected point
            cv2.circle(roi, (int(dart_pixel_x), roi_center_y), 5, (0, 255, 0), -1)
            cv2.putText(roi, f"Px: {dart_pixel_x:.2f}", (int(dart_pixel_x) + 5, roi_center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw the contour
            cv2.drawContours(roi, [best_contour], 0, (0, 255, 0), 2)
            
            # In calibration mode, display the pixel values
            if self.calibration_mode and self.calibration_point:
                print(f"Cam1 pixel for {self.calibration_point}: {dart_pixel_x:.2f}")
            
            # Get board x-coordinate using interpolation
            if self.use_advanced_interpolation:
                board_x = None  # Will be determined in the final step with both camera values
            else:
                # Use standard interpolation
                board_x = self.interpolate_value(dart_pixel_x, self.cam1_pixel_to_board_mapping)
                
                # Apply smoothing
                board_x = self.apply_smoothing(board_x, 'cam1')
                
                # Store vector info
                self.cam1_vector = (board_x, 0)
                
                # Display board coordinates
                cv2.putText(roi, f"Board X: {board_x:.1f}mm", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None
        
        # Copy ROI back to rotated frame
        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def process_camera2_frame(self, frame):
        """Process camera 2 frame with enhanced detection"""
        # Rotate frame 180 degrees as in original code
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor2.apply(blurred)
        
        # Threshold and clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to improve mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        dart_pixel_x = None
        best_contour = None
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        # Process contours
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                # Use enhanced subpixel detection
                dart_pixel_x = self.detect_point_with_subpixel(contour, roi_center_y)
                if dart_pixel_x is not None:
                    best_contour = contour
                    break
        
        # If dart detected
        if dart_pixel_x is not None:
            # Highlight the detected point
            cv2.circle(roi, (int(dart_pixel_x), roi_center_y), 5, (0, 255, 0), -1)
            cv2.putText(roi, f"Px: {dart_pixel_x:.2f}", (int(dart_pixel_x) + 5, roi_center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw the contour
            cv2.drawContours(roi, [best_contour], 0, (0, 255, 0), 2)
            
            # In calibration mode, display the pixel values
            if self.calibration_mode and self.calibration_point:
                print(f"Cam2 pixel for {self.calibration_point}: {dart_pixel_x:.2f}")
            
            # Get board y-coordinate using interpolation
            if self.use_advanced_interpolation:
                board_y = None  # Will be determined in the final step with both camera values
            else:
                # Use standard interpolation
                board_y = self.interpolate_value(dart_pixel_x, self.cam2_pixel_to_board_mapping)
                
                # Apply smoothing
                board_y = self.apply_smoothing(board_y, 'cam2')
                
                # Store vector info
                self.cam2_vector = (0, board_y)
                
                # Display board coordinates
                cv2.putText(roi, f"Board Y: {board_y:.1f}mm", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
        
       
# Copy ROI back to rotated frame
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def compute_intersection(self):
        """
        Compute intersection of the vectors from both cameras.
        Enhanced version with additional validation.
        """
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
            
        # For cam1, we've determined the board_x value
        cam1_board_x = self.cam1_vector[0]
        cam1_ray_start = self.camera1_position
        cam1_ray_end = (cam1_board_x, 0)
        
        # For cam2, we've determined the board_y value
        cam2_board_y = self.cam2_vector[1]
        cam2_ray_start = self.camera2_position
        cam2_ray_end = (0, cam2_board_y)
        
        # Find the intersection of these rays
        intersection = self.compute_line_intersection(
            cam1_ray_start, cam1_ray_end,
            cam2_ray_start, cam2_ray_end
        )
        
        # Validate intersection is within reasonable bounds of the board
        if intersection:
            x, y = intersection
            distance_from_center = math.sqrt(x*x + y*y)
            
            # Check if the point is within a reasonable distance from the board
            if distance_from_center <= self.board_radius + 10:  # 10mm margin
                return intersection
        
        return None
    
    def compute_line_intersection(self, p1, p2, p3, p4):
        """
        Compute the intersection of two lines.
        Enhanced version for better numerical stability.
        """
        # Convert inputs to numpy arrays for better handling
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        p3 = np.array(p3, dtype=np.float64)
        p4 = np.array(p4, dtype=np.float64)
        
        # Calculate direction vectors
        v1 = p2 - p1
        v2 = p4 - p3
        
        # Calculate cross product to check if lines are parallel
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        # If lines are parallel (or nearly parallel)
        if abs(cross_product) < 1e-10:
            return None
        
        # Calculate intersection parameter for first line
        v3 = p1 - p3
        t = (v2[0] * v3[1] - v2[1] * v3[0]) / cross_product
        
        # Calculate intersection point
        intersection = p1 + t * v1
        
        return tuple(intersection)
    
    def calculate_score(self, position):
        """
        Calculate score based on dart position
        
        Args:
            position: (x, y) position in board coordinates
            
        Returns:
            score: Score value
            description: Text description of the hit
        """
        if position is None:
            return 0, "No hit"
            
        x, y = position
        distance_from_center = math.sqrt(x*x + y*y)
        
        # Check if dart is outside the board
        if distance_from_center > self.board_radius:
            return 0, "Outside board"
            
        # Check bullseye
        if distance_from_center <= 12.7:  # Inner bullseye
            return 50, "Bullseye (50)"
        elif distance_from_center <= 31.8:  # Outer bullseye
            return 25, "Outer bull (25)"
            
        # Determine segment number based on angle
        angle_rad = math.atan2(y, x)
        angle_deg = math.degrees(angle_rad)
        
        # Convert to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
            
        # Calculate segment index (20 segments, starting from top and going clockwise)
        segment_angle = (450 - angle_deg) % 360
        segment_index = int(segment_angle / 18)  # Each segment is 18 degrees
        
        # Map to dartboard segment numbers
        segment_map = [
            20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
        ]
        
        segment_number = segment_map[segment_index % 20]
        
        # Determine multiplier based on distance
        if 169 <= distance_from_center <= 171:  # Double ring
            multiplier = 2
            hit_type = "Double"
        elif 105 <= distance_from_center <= 107:  # Triple ring
            multiplier = 3
            hit_type = "Triple"
        else:  # Single
            multiplier = 1
            hit_type = "Single"
            
        score = segment_number * multiplier
        
        if multiplier > 1:
            description = f"{hit_type} {segment_number} ({score})"
        else:
            description = f"{segment_number}"
            
        return score, description
    
    def update_board_projection(self, cam1_pixel_x=None, cam2_pixel_x=None):
        """
        Update board projection with current dart position
        Enhanced version with improved visualization and accuracy
        
        Args:
            cam1_pixel_x, cam2_pixel_x: Raw pixel values from cameras (optional)
        """
        # Create a larger canvas to display the board and camera positions
        canvas_size = 600
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Calculate the scale factor to fit the board properly
        board_px_radius = 200
        scale_factor = board_px_radius / self.board_radius
        
        # Calculate center of the canvas
        canvas_center_x = canvas_size // 2
        canvas_center_y = canvas_size // 2
        
        # Function to convert from mm coordinates to canvas pixel coordinates
        def mm_to_canvas_px(x, y):
            px = int(canvas_center_x + x * scale_factor)
            py = int(canvas_center_y - y * scale_factor)
            return (px, py)
        
        # Draw the board image
        if self.board_image is not None:
            # Calculate size to make the dartboard fill the boundary circle
            board_size = int(self.board_radius * 2 * scale_factor)
            image_scale_multiplier = 2.75  # Adjust to make dartboard fill boundary
            board_img_size = int(board_size * image_scale_multiplier)
            
            board_resized = cv2.resize(self.board_image, (board_img_size, board_img_size))
            
            # Calculate position to paste the board image (centered)
            board_x = canvas_center_x - board_img_size // 2
            board_y = canvas_center_y - board_img_size // 2
            
            # Create circular mask
            mask = np.zeros((board_img_size, board_img_size), dtype=np.uint8)
            cv2.circle(mask, (board_img_size//2, board_img_size//2), board_img_size//2, 255, -1)
            
            # Paste board image with mask
            if (board_x >= 0 and board_y >= 0 and 
                board_x + board_img_size <= canvas_size and 
                board_y + board_img_size <= canvas_size):
                canvas_roi = canvas[board_y:board_y+board_img_size, board_x:board_x+board_img_size]
                board_masked = cv2.bitwise_and(board_resized, board_resized, mask=mask)
                canvas_roi[mask > 0] = board_masked[mask > 0]
        
        # Draw reference grid and board circles
        cv2.line(canvas, (0, canvas_center_y), (canvas_size, canvas_center_y), (200, 200, 200), 1)
        cv2.line(canvas, (canvas_center_x, 0), (canvas_center_x, canvas_size), (200, 200, 200), 1)
        
        # Draw board boundary and rings
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(self.board_radius * scale_factor), (0, 0, 0), 1)
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(107 * scale_factor), (0, 0, 0), 1)  # Triple ring
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(170 * scale_factor), (0, 0, 0), 1)  # Double ring
        
        # Draw bullseye rings
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(12.7 * scale_factor), (0, 0, 0), 1)  # Inner bull
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(31.8 * scale_factor), (0, 0, 0), 1)  # Outer bull
        
        # Draw segment markers
        for segment, (x, y) in self.board_segments.items():
            segment_px = mm_to_canvas_px(x, y)
            cv2.circle(canvas, segment_px, 3, (128, 0, 128), -1)
            cv2.putText(canvas, f"{segment}", (segment_px[0]+5, segment_px[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
        
        # Draw camera positions
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        
        cv2.circle(canvas, cam1_px, 8, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(canvas, cam2_px, 8, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Advanced interpolation when both cameras have detections
        if cam1_pixel_x is not None and cam2_pixel_x is not None and self.use_advanced_interpolation:
            # Perform advanced interpolation
            board_x, board_y = self.advanced_interpolation(cam1_pixel_x, cam2_pixel_x)
            
            # Apply Kalman filtering for smooth tracking
            if board_x is not None and board_y is not None:
                # Update Kalman filter
                filtered_position = self.update_kalman((board_x, board_y))
                
                if filtered_position is not None:
                    # Apply additional smoothing if needed
                    smoothed_position = self.apply_smoothing(filtered_position, 'final')
                    
                    if smoothed_position is not None:
                        # Store final position
                        self.final_tip = smoothed_position
                        
                        # Check for stable position
                        if self.last_stable_position is not None:
                            # Calculate movement from last stable position
                            dx = smoothed_position[0] - self.last_stable_position[0]
                            dy = smoothed_position[1] - self.last_stable_position[1]
                            movement = math.sqrt(dx*dx + dy*dy)
                            
                            # If movement is below threshold, increment stable counter
                            if movement < self.movement_threshold:
                                self.stable_position_count += 1
                            else:
                                self.stable_position_count = 0
                        
                        # Update last stable position
                        self.last_stable_position = smoothed_position
                        
                        # Draw the dart position on board
                        dart_px = mm_to_canvas_px(*smoothed_position)
                        
                        # Color based on stability
                        if self.stable_position_count >= 10:
                            # Stable position (yellow)
                            color = (0, 255, 255)
                        else:
                            # Moving position (green)
                            color = (0, 255, 0)
                        
                        # Draw dart position
                        cv2.circle(canvas, dart_px, 8, (0, 0, 0), -1)  # Black outline
                        cv2.circle(canvas, dart_px, 6, color, -1)  # Colored center
                        
                        # Display coordinates
                        x, y = smoothed_position
                        cv2.putText(canvas, f"({x:.1f}, {y:.1f})", (dart_px[0]+10, dart_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(canvas, f"({x:.1f}, {y:.1f})", (dart_px[0]+10, dart_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Calculate and display score
                        score, description = self.calculate_score(smoothed_position)
                        
                        score_text = f"Score: {score} - {description}"
                        cv2.putText(canvas, score_text, (dart_px[0]+10, dart_px[1]+25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(canvas, score_text, (dart_px[0]+10, dart_px[1]+25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # If position is stable for a while, display "STABLE" indicator
                        if self.stable_position_count >= 10:
                            cv2.putText(canvas, "STABLE", (dart_px[0]-20, dart_px[1]-20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Standard processing from original code
            # --- Cam1 vector ---
            if self.cam1_vector is not None:
                board_point = mm_to_canvas_px(*self.cam1_vector)
                cv2.circle(canvas, board_point, 5, (0, 0, 255), -1)
                cv2.putText(canvas, f"X: {self.cam1_vector[0]:.1f}", (board_point[0]+5, board_point[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw vector from camera
                cv2.line(canvas, cam1_px, board_point, (0, 0, 255), 2)
                
                # Calculate and draw extended vector
                dx = board_point[0] - cam1_px[0]
                dy = board_point[1] - cam1_px[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    extension_factor = 2.0
                    extended_x = int(board_point[0] + dx * extension_factor)
                    extended_y = int(board_point[1] + dy * extension_factor)
                    extended_pt = (extended_x, extended_y)
                    cv2.line(canvas, board_point, extended_pt, (0, 0, 255), 2)
            
            # --- Cam2 vector ---
            if self.cam2_vector is not None:
                board_point = mm_to_canvas_px(*self.cam2_vector)
                cv2.circle(canvas, board_point, 5, (255, 0, 0), -1)
                cv2.putText(canvas, f"Y: {self.cam2_vector[1]:.1f}", (board_point[0]+5, board_point[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw vector from camera
                cv2.line(canvas, cam2_px, board_point, (255, 0, 0), 2)
                
                # Calculate and draw extended vector
                dx = board_point[0] - cam2_px[0]
                dy = board_point[1] - cam2_px[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    extension_factor = 2.0
                    extended_x = int(board_point[0] + dx * extension_factor)
                    extended_y = int(board_point[1] + dy * extension_factor)
                    extended_pt = (extended_x, extended_y)
                    cv2.line(canvas, board_point, extended_pt, (255, 0, 0), 2)
            
            # --- Final Dart Position from vector intersection ---
            if self.cam1_vector is not None and self.cam2_vector is not None:
                # Calculate intersection
                self.final_tip = self.compute_intersection()
                
                if self.final_tip is not None:
                    # Apply smoothing
                    smoothed_final_tip = self.apply_smoothing(self.final_tip, 'final')
                    
                    if smoothed_final_tip:
                        dart_x, dart_y = smoothed_final_tip
                        final_px = mm_to_canvas_px(dart_x, dart_y)
                        
                        # Draw intersection point
                        cv2.circle(canvas, final_px, 8, (0, 0, 0), -1)  # Black outline
                        cv2.circle(canvas, final_px, 6, (0, 255, 0), -1)  # Green center
                        
                        # Calculate score and description
                        score, description = self.calculate_score(smoothed_final_tip)
                        
                        # Display position and score
                        label = f"Dart: ({dart_x:.1f}, {dart_y:.1f}) - {description}"
                        
                        cv2.putText(canvas, label, (final_px[0]+10, final_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(canvas, label, (final_px[0]+10, final_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        
                        # Print hit information to terminal
                        print(f"\nDart hit at ({dart_x:.1f}, {dart_y:.1f}) mm")
                        print(f"Distance from center: {math.sqrt(dart_x**2 + dart_y**2):.1f} mm")
                        print(f"Score: {score} - {description}")
        
        # Add calibration mode indicator if active
        if self.calibration_mode:
            cv2.putText(canvas, "CALIBRATION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.calibration_point:
                cv2.putText(canvas, f"Current point: {self.calibration_point}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Highlight current calibration point on board
                cal_px = mm_to_canvas_px(*self.calibration_point)
                cv2.circle(canvas, cal_px, 10, (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]-15), (cal_px[0]+15, cal_px[1]+15), (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]+15), (cal_px[0]+15, cal_px[1]-15), (0, 0, 255), 2)
        
        # Add FPS counter
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (10, canvas_size-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                  
        # Add accuracy indicator (1mm target)
        cv2.putText(canvas, "Target accuracy: 1mm", (canvas_size-250, canvas_size-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return canvas
    
    def toggle_calibration_mode(self):
        """Toggle calibration mode on/off"""
        self.calibration_mode = not self.calibration_mode
        if self.calibration_mode:
            print("\n*** CALIBRATION MODE ACTIVATED ***")
            print("Place dart at known positions and press 'c' to capture pixel values")
            print("Press 't' to toggle calibration mode off when done")
        else:
            print("\n*** CALIBRATION MODE DEACTIVATED ***")
            self.calibration_point = None
    
    def set_calibration_point(self, board_x, board_y):
        """Set the current calibration point coordinates"""
        self.calibration_point = (board_x, board_y)
        print(f"\nCalibration point set to ({board_x}, {board_y})")
        print("Place dart at this position and press 'c' to capture pixel values")
    
    def optimize_existing_calibration(self):
        """
        Optimize the existing calibration data by:
        1. Adding interpolated points
        2. Refining calibration with polynomial fitting
        3. Validating accuracy using cross-validation
        """
        print("\nOptimizing existing calibration...")
        
        # Create manual calibration enhancer
        from manual_calibration import ManualCalibrationEnhancer
        enhancer = ManualCalibrationEnhancer(self.calibration_points)
        
        # Generate enhanced calibration points
        enhancer.generate_enhanced_calibration()
        
        # Update our calibration points with enhanced ones
        self.calibration_points = enhancer.calibration_points
        
        # Rebuild mapping tables
        self.cam1_pixel_to_board_mapping = []
        self.cam2_pixel_to_board_mapping = []
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
        
        print(f"Calibration optimized - now using {len(self.calibration_points)} points")
    
    def run(self):
        """Run the dart tracking system with optimized settings for 1mm accuracy"""
        # Initialize cameras
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Try to load calibration data
        try:
            self.load_calibration()
        except:
            print("Using default calibration data")
        
        # Print instructions
        print("\n*** OPTIMIZED DART TRACKING SYSTEM ***")
        print("Target accuracy: 1mm")
        print("Press 'q' to exit")
        print("Press 't' to toggle calibration mode")
        print("Press 'c' in calibration mode to capture current point")
        print("Press 'r' to reset background subtractors")
        print("Press 's' to save current calibration to file")
        print("Press 'l' to load calibration from file")
        print("Press 'o' to optimize existing calibration (for 1mm accuracy)")
        
        # FPS calculation variables
        frame_count = 0
        fps_start_time = time.time()
        
        # Main loop
        while True:
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Error reading from cameras")
                break
            
            # Process frames
            proc_frame1, fg_mask1, cam1_pixel_x = self.process_camera1_frame(frame1)
            proc_frame2, fg_mask2, cam2_pixel_x = self.process_camera2_frame(frame2)
            
            # Update board projection
            board_proj = self.update_board_projection(cam1_pixel_x, cam2_pixel_x)
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - fps_start_time
            
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = current_time
            
            # Display frames
            cv2.imshow("Camera 1", proc_frame1)
            cv2.imshow("Camera 2", proc_frame2)
            cv2.imshow("Board", board_proj)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.toggle_calibration_mode()
            elif key == ord('c') and self.calibration_mode:
                if self.calibration_point is None:
                    print("Please set calibration point first (using number keys 0-9)")
                else:
                    # Add the current detection to calibration points
                    if cam1_pixel_x is not None and cam2_pixel_x is not None:
                        print(f"Adding calibration point: {self.calibration_point} -> "
                             f"Cam1: {cam1_pixel_x:.2f}, Cam2: {cam2_pixel_x:.2f}")
                        
                        self.calibration_points[self.calibration_point] = (
                            int(round(cam1_pixel_x)), 
                            int(round(cam2_pixel_x))
                        )
                        
                        # Rebuild mapping tables
                        self.cam1_pixel_to_board_mapping = []
                        self.cam2_pixel_to_board_mapping = []
                        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
                            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
                            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
                        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
                        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
                    else:
                        print("Could not detect dart in one or both cameras")
            elif key == ord('r'):
                print("Resetting background subtractors")
                self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
                    history=3000, varThreshold=60, detectShadows=False
                )
                self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
                    history=3000, varThreshold=60, detectShadows=False
                )
                # Reset Kalman filter
                self.kalman_initialized = False
                # Reset stable position tracking
                self.stable_position_count = 0
                self.last_stable_position = None
            elif key == ord('s'):
                print("Saving calibration to file")
                self.save_calibration()
            elif key == ord('l'):
                print("Loading calibration from file")
                self.load_calibration()
            elif key == ord('o'):
                self.optimize_existing_calibration()
            elif key >= ord('0') and key <= ord('9') and self.calibration_mode:
                # Quick set calibration point using number keys
                segment_num = key - ord('0')
                if segment_num in self.board_segments:
                    self.set_calibration_point(*self.board_segments[segment_num])
                else:
                    print(f"No segment {segment_num} defined")
        
        # Cleanup
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = OptimizedDartTracker()
    tracker.run()
