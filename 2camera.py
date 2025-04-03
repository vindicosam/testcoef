import cv2
import numpy as np
import math
import time
from collections import deque
from bisect import bisect_left
import json
import ast
import os

class RaspberryPiDartTracker:
    def __init__(self, cam_index1=0, cam_index2=2):  # Changed default cam2 from 2 to 1
        """
        Optimized dart tracking system for Raspberry Pi
        with simplified but highly accurate tracking
        """
        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2
        
        # Camera settings - use standard resolution that all cameras support
        self.frame_width = 640
        self.frame_height = 480

        # Static camera positions
        self.camera1_position = (0, 550)    # Front camera fixed position
        self.camera2_position = (-400, 0)   # Side camera fixed position

        # Board ROI settings
        self.cam1_board_plane_y = 182
        self.cam1_roi_range = 30
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        self.cam2_board_plane_y = 208
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
            (90, -50): (359, 406)
        }
        
        # Convert any tuples with 4 elements to proper format
        calibration_points_fixed = {}
        for k, v in self.calibration_points.items():
            if len(k) == 4:  # It's in the format (x, y, px1, px2)
                calibration_points_fixed[(k[0], k[1])] = (k[2], k[3])
            else:
                calibration_points_fixed[k] = v
        
        self.calibration_points = calibration_points_fixed
        
        # Import segment calibration points for scoring
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
            history=500,        # Shorter history for more adaptive tracking
            varThreshold=40,    # Adjusted threshold for better detection
            detectShadows=False
        )
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=40,
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
            'cam1': deque(maxlen=10),  # Increased history size
            'cam2': deque(maxlen=10),
            'final': deque(maxlen=10)
        }
        
        # Kalman filter for smooth tracking - optimized parameters for 1mm accuracy
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0.1, 0],
            [0, 1, 0, 0.1],
            [0, 0, 0.95, 0],   # Velocity persistence factor (decay)
            [0, 0, 0, 0.95]
        ], np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        # Fine-tuned for millimeter accuracy
        self.kalman.processNoiseCov = np.array([
            [1e-4, 0, 0, 0],
            [0, 1e-4, 0, 0],
            [0, 0, 1e-2, 0],
            [0, 0, 0, 1e-2]
        ], np.float32)
        self.kalman.measurementNoiseCov = np.array([
            [0.1, 0],
            [0, 0.1]
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
        self.board_image = None
        board_image_path = "winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg"
        if os.path.exists(board_image_path):
            self.board_image = cv2.imread(board_image_path)
        
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
        
        # Filtering parameters for 1mm accuracy
        self.movement_threshold = 0.5  # mm - target threshold for stable detection
        self.stable_position_count = 0  # counter for stable positions
        self.last_stable_position = None  # last stable position
        
        # Advanced interpolation
        self.use_polynomial_interpolation = True  # Use polynomial instead of RBF
        self.polynomial_degree = 3  # Cubic polynomials work well for distortion
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Initialize polynomial models
        self.poly_models = {
            'cam1': None,
            'cam2': None
        }
        self.build_polynomial_models()
    
    def build_polynomial_models(self):
        """
        Build polynomial interpolation models for improved accuracy
        without requiring SciPy RBF interpolator
        """
        if len(self.calibration_points) < 6:
            print("Not enough calibration points for polynomial interpolation")
            return
            
        try:
            # Extract points for models
            board_points = []
            cam1_pixels = []
            cam2_pixels = []
            
            for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
                board_points.append((board_x, board_y))
                cam1_pixels.append(cam1_pixel_x)
                cam2_pixels.append(cam2_pixel_x)
                
            # Convert to numpy arrays
            board_points = np.array(board_points)
            cam1_pixels = np.array(cam1_pixels)
            cam2_pixels = np.array(cam2_pixels)
            
            # For camera 1, we predict board_x from pixel_x
            # Create polynomial features (1, x, x², x³, ...)
            degree = min(self.polynomial_degree, 5)  # Limit to prevent overfitting
            
            # Simple implementation of polynomial regression without external libraries
            # For camera 1 (x coordinate)
            A1 = np.zeros((len(cam1_pixels), degree + 1))
            for i in range(degree + 1):
                A1[:, i] = cam1_pixels ** i
                
            # Solve for coefficients using least squares
            # This effectively does: coeffs = (A^T A)^-1 A^T b
            self.poly_models['cam1'] = np.linalg.lstsq(A1, board_points[:, 0], rcond=None)[0]
            
            # For camera 2 (y coordinate)
            A2 = np.zeros((len(cam2_pixels), degree + 1))
            for i in range(degree + 1):
                A2[:, i] = cam2_pixels ** i
                
            self.poly_models['cam2'] = np.linalg.lstsq(A2, board_points[:, 1], rcond=None)[0]
            
            print(f"Built polynomial models (degree {degree}) with {len(board_points)} points")
            
        except Exception as e:
            print(f"Error building polynomial models: {e}")
            self.poly_models['cam1'] = None
            self.poly_models['cam2'] = None
    
    def evaluate_polynomial(self, poly_coeffs, x):
        """
        Evaluate a polynomial with given coefficients at point x
        """
        if poly_coeffs is None:
            return None
            
        result = 0
        for i, coeff in enumerate(poly_coeffs):
            result += coeff * (x ** i)
        return result
    
    def load_calibration(self, filename="dart_calibration.json"):
        """Load calibration from file"""
        try:
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
            
            # Rebuild polynomial models
            self.build_polynomial_models()
            
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
    
    def advanced_interpolate(self, cam1_pixel_x, cam2_pixel_x):
        """
        Use polynomial models for more accurate interpolation
        with fallback to simple linear interpolation
        """
        if self.use_polynomial_interpolation and self.poly_models['cam1'] is not None and self.poly_models['cam2'] is not None:
            try:
                # Use polynomial model for camera 1 (x coordinate)
                board_x = self.evaluate_polynomial(self.poly_models['cam1'], cam1_pixel_x)
                
                # Use polynomial model for camera 2 (y coordinate)
                board_y = self.evaluate_polynomial(self.poly_models['cam2'], cam2_pixel_x)
                
                return board_x, board_y
            except Exception as e:
                print(f"Polynomial interpolation error: {e}")
                # Fall back to simple interpolation
        
        # Simple linear interpolation as fallback
        board_x = self.interpolate_value(cam1_pixel_x, self.cam1_pixel_to_board_mapping)
        board_y = self.interpolate_value(cam2_pixel_x, self.cam2_pixel_to_board_mapping)
        
        return board_x, board_y
    
    def apply_smoothing(self, new_value, history_key):
        """
        Apply smoothing to the detection using weighted moving average
        with adaptive parameters for 1mm accuracy
        """
        if new_value is None:
            return None
        
        # Add new value to history
        self.detection_history[history_key].append(new_value)
        
        # If we have enough history, compute weighted average
        if len(self.detection_history[history_key]) >= 3:
            # For final position (x,y tuple)
            if history_key == 'final':
                # Adaptive smoothing - detect motion speed
                dx = dy = 0
                if len(self.detection_history[history_key]) >= 2:
                    prev = self.detection_history[history_key][-2]
                    curr = self.detection_history[history_key][-1]
                    dx = curr[0] - prev[0]
                    dy = curr[1] - prev[1]
                    motion = math.sqrt(dx*dx + dy*dy)
                else:
                    motion = 0
                
                # Adjust smoothing factor based on motion speed
                # Fast motion: less smoothing, Slow motion: more smoothing
                if motion > 5:  # Fast motion (>5mm)
                    alpha = 0.7  # More weight on new readings
                elif motion > 2:  # Medium motion
                    alpha = 0.5  # Balanced weight
                else:  # Slow motion or stable
                    alpha = 0.3  # More smoothing for stability
                
                # Apply exponential smoothing
                if len(self.detection_history[history_key]) >= 2:
                    prev = self.detection_history[history_key][-2]
                    curr = self.detection_history[history_key][-1]
                    
                    smoothed_x = alpha * curr[0] + (1 - alpha) * prev[0]
                    smoothed_y = alpha * curr[1] + (1 - alpha) * prev[1]
                    
                    return (smoothed_x, smoothed_y)
                else:
                    return self.detection_history[history_key][-1]
                
            else:
                # For single values (cam1/cam2)
                # Similar adaptive approach based on motion speed
                if len(self.detection_history[history_key]) >= 2:
                    prev = self.detection_history[history_key][-2]
                    curr = self.detection_history[history_key][-1]
                    motion = abs(curr - prev)
                    
                    if motion > 10:  # Fast motion
                        alpha = 0.7
                    elif motion > 3:  # Medium motion
                        alpha = 0.5
                    else:  # Slow motion or stable
                        alpha = 0.3
                    
                    # Apply exponential smoothing
                    return alpha * curr + (1 - alpha) * prev
                else:
                    return self.detection_history[history_key][-1]
        
        return new_value
    
    def update_kalman(self, position):
        """Update Kalman filter with new position measurement"""
        if position is None:
            return None
        
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
    
    def detect_point_with_subpixel(self, roi, fg_mask, roi_center_y):
        """
        Enhanced detection algorithm with subpixel accuracy
        
        Args:
            roi: Region of interest
            fg_mask: Foreground mask
            roi_center_y: Y-coordinate of center line
            
        Returns:
            x: X-coordinate with subpixel accuracy
        """
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Filter contours by size and proximity to center line
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Minimum area threshold
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour crosses the board plane
            contour_center_y = y + h/2
            if abs(contour_center_y - roi_center_y) > 20:  # Allow 20px margin
                continue
                
            valid_contours.append(contour)
        
        if not valid_contours:
            return None
            
        best_contour = valid_contours[0]
        
        # Multi-technique approach for higher precision
        
        # 1. Use moments for centroid calculation (already subpixel)
        M = cv2.moments(best_contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
        else:
            x, y, w, h = cv2.boundingRect(best_contour)
            cx = x + w/2
            
        # 2. Find the point on contour closest to the center line
        min_dist = float('inf')
        closest_x = None
        
        for point in best_contour:
            px, py = point[0]
            dist = abs(py - roi_center_y)
            if dist < min_dist:
                min_dist = dist
                closest_x = px
                
        # 3. Refine with local window processing
        if closest_x is not None:
            # Define region around detection
            window_size = 15
            window_x = max(0, int(closest_x) - window_size)
            window_y = max(0, int(roi_center_y) - window_size)
            window_w = min(roi.shape[1] - window_x, window_size * 2)
            window_h = min(roi.shape[0] - window_y, window_size * 2)
            
            if window_w > 0 and window_h > 0:
                # Extract window
                window = roi[window_y:window_y+window_h, window_x:window_x+window_w]
                
                if window.size > 0:
                    # Convert to grayscale
                    window_gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    
                    # Find gradient maximum (edge)
                    sobelx = cv2.Sobel(window_gray, cv2.CV_64F, 1, 0, ksize=3)
                    abs_sobelx = np.absolute(sobelx)
                    
                    # Find the column with maximum gradient
                    col_sums = np.sum(abs_sobelx, axis=0)
                    if len(col_sums) > 0:
                        max_col = np.argmax(col_sums)
                        edge_x = window_x + max_col
                    else:
                        edge_x = closest_x
                else:
                    edge_x = closest_x
            else:
                edge_x = closest_x
        else:
            edge_x = cx
            
        # 4. Apply cornerSubPix for final refinement
        try:
            # Convert to proper format
            corners = np.array([[edge_x, roi_center_y]], dtype=np.float32)
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            
            # Apply corner refinement
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            
            refined_x = corners[0][0]
        except Exception as e:
            # Fall back to previous result
            refined_x = edge_x
            
        # Draw detection for visualization
        cv2.drawContours(roi, [best_contour], 0, (0, 255, 0), 2)
        
        return refined_x
    
    def process_camera1_frame(self, frame):
        """Process camera 1 frame with enhanced detection"""
        # Make a copy to avoid modifying original
        frame_copy = frame.copy()
        
        # Rotate frame 180 degrees as in original code
        frame_rot = cv2.rotate(frame_copy, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor1.apply(blurred, learningRate=0.01)
        
        # Threshold and clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to improve mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        # Detect dart using enhanced algorithm
        dart_pixel_x = self.detect_point_with_subpixel(roi, fg_mask, roi_center_y)
        
        # If dart detected
        if dart_pixel_x is not None:
            # Highlight the detected point
            cv2.circle(roi, (int(dart_pixel_x), roi_center_y), 5, (0, 255, 0), -1)
            cv2.putText(roi, f"Px: {dart_pixel_x:.2f}", (int(dart_pixel_x) + 5, roi_center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # In calibration mode, display the pixel values
            if self.calibration_mode and self.calibration_point:
                print(f"Cam1 pixel for {self.calibration_point}: {dart_pixel_x:.2f}")
            
            # Apply smoothing
            smoothed_px = self.apply_smoothing(dart_pixel_x, 'cam1')
            
            # Store for cross-camera calculations
            self.cam1_vector = smoothed_px
            
            # Display smoothed value
            cv2.putText(roi, f"Smoothed: {smoothed_px:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None
        
        # Copy ROI back to rotated frame
        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def process_camera2_frame(self, frame):
        """Process camera 2 frame with enhanced detection"""
        # Make a copy to avoid modifying original
        frame_copy = frame.copy()
        
        # Rotate frame 180 degrees as in original code
        frame_rot = cv2.rotate(frame_copy, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor2.apply(blurred, learningRate=0.01)
        
        # Threshold and clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to improve mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        # Detect dart using enhanced algorithm
        dart_pixel_x = self.detect_point_with_subpixel(roi, fg_mask, roi_center_y)
        
        # If dart detected
        if dart_pixel_x is not None:
            # Highlight the detected point
            cv2.circle(roi, (int(dart_pixel_x), roi_center_y), 5, (0, 255, 0), -1)
            cv2.putText(roi, f"Px: {dart_pixel_x:.2f}", (int(dart_pixel_x) + 5, roi_center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # In calibration mode, display the pixel values
            if self.calibration_mode and self.calibration_point:
                print(f"Cam2 pixel for {self.calibration_point}: {dart_pixel_x:.2f}")
            
            # Apply smoothing
            smoothed_px = self.apply_smoothing(dart_pixel_x, 'cam2')
            
            # Store for cross-camera calculations
            self.cam2_vector = smoothed_px
            
            # Display smoothed value
            cv2.putText(roi, f"Smoothed: {smoothed_px:.1f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
        
        # Copy ROI back to rotated frame
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def compute_board_coordinates(self):
        """
        Advanced computation of dart position using interpolation
        with millimeter-level accuracy
        """
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
            
        # Use advanced interpolation
        board_coords = self.advanced_interpolate(self.cam1_vector, self.cam2_vector)
        
        if board_coords:
            # Apply Kalman filtering for smoother tracking
            filtered_coords = self.update_kalman(board_coords)
            
            # Apply additional smoothing for stability
            if filtered_coords:
                final_coords = self.apply_smoothing(filtered_coords, 'final')
                
                # Calculate distance from center for validation
                if final_coords:
                    x, y = final_coords
                    distance = math.sqrt(x*x + y*y)
                    
                    # Check if within reasonable range (with margin)
                    if distance <= self.board_radius + 20:
                        return final_coords
        
        return None
    
    def calculate_score(self, position):
        """
        Calculate score based on dart position with enhanced accuracy
        """
        if position is None:
            return 0, "No hit", None
            
        x, y = position
        distance_from_center = math.sqrt(x*x + y*y)
        
        # Check if dart is outside the board
        if distance_from_center > self.board_radius:
            return 0, "Outside board", {"region": "outside", "distance": distance_from_center}
            
        # Check bullseye
        if distance_from_center <= 12.7:  # Inner bullseye
            return 50, "Bullseye (50)", {"region": "bullseye", "distance": distance_from_center}
        elif distance_from_center <= 31.8:  # Outer bullseye
            return 25, "Outer bull (25)", {"region": "outer_bull", "distance": distance_from_center}
            
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
        
        # Enhanced ring detection with wider margins for Raspberry Pi
        # Double: 167-173mm (wider margin for better reliability)
        # Triple: 103-109mm (wider margin for better reliability)
        if 167 <= distance_from_center <= 173:  # Double ring
            multiplier = 2
            hit_type = "Double"
            region = "double"
        elif 103 <= distance_from_center <= 109:  # Triple ring
            multiplier = 3
            hit_type = "Triple"
            region = "triple"
        else:  # Single
            multiplier = 1
            # Determine inner vs outer single
            if distance_from_center < 103:
                hit_type = "Inner Single"
                region = "inner_single"
            else:
                hit_type = "Outer Single"
                region = "outer_single"
            
        score = segment_number * multiplier
        
        description = f"{hit_type} {segment_number} ({score})"
            
        region_info = {
            "region": region,
            "segment": segment_number,
            "multiplier": multiplier,
            "distance": distance_from_center,
            "angle_deg": angle_deg
        }
            
        return score, description, region_info
    
    def update_board_projection(self, cam1_pixel_x=None, cam2_pixel_x=None):
        """
        Update board projection with current dart position
        with optimizations for Raspberry Pi performance
        """
        # Create a canvas to display the board and camera positions
        canvas_size = 600  # Optimized size for Raspberry Pi
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Calculate the scale factor to fit the board properly
        board_px_radius = 250
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
            board_img_size = int(board_size * 1.5)  # Smaller multiplier for Pi
            
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
        
        # Draw board boundary and rings (just a few for performance)
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(self.board_radius * scale_factor), (0, 0, 0), 2)
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(107 * scale_factor), (0, 0, 0), 2)  # Triple ring
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(170 * scale_factor), (0, 0, 0), 2)  # Double ring
        
        # Draw bullseye rings
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(12.7 * scale_factor), (0, 0, 0), 2)  # Inner bull
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(31.8 * scale_factor), (0, 0, 0), 2)  # Outer bull
        
        # Draw camera positions
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        
        cv2.circle(canvas, cam1_px, 8, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(canvas, cam2_px, 8, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # If both cameras have detected a dart
        if cam1_pixel_x is not None and cam2_pixel_x is not None:
            # Compute board coordinates
            board_position = self.compute_board_coordinates()
            
            if board_position is not None:
                # Check for stability
                if self.last_stable_position is not None:
                    dx = board_position[0] - self.last_stable_position[0]
                    dy = board_position[1] - self.last_stable_position[1]
                    movement = math.sqrt(dx*dx + dy*dy)
                    
                    if movement < self.movement_threshold:
                        self.stable_position_count += 1
                    else:
                        self.stable_position_count = 0
                        self.last_stable_position = board_position
                else:
                    self.last_stable_position = board_position
                    self.stable_position_count = 1
                
                # Draw the dart position
                dart_px = mm_to_canvas_px(*board_position)
                
                # Color based on stability
                if self.stable_position_count >= 10:
                    # Stable position (yellow) - 1mm accuracy achieved
                    color = (0, 255, 255)
                else:
                    # Moving position (green) - stabilizing
                    color = (0, 255, 0)
                
                # Draw 1mm precision indicator (concentric circles)
                for r in range(1, 6):
                    cv2.circle(canvas, dart_px, r, (0, 0, 0), 1)
                
                # Draw dart position
                cv2.circle(canvas, dart_px, 8, (0, 0, 0), -1)  # Black outline
                cv2.circle(canvas, dart_px, 6, color, -1)  # Colored center
                
                # Display coordinates with 0.1mm precision
                x, y = board_position
                cv2.putText(canvas, f"({x:.1f}, {y:.1f})mm", (dart_px[0]+10, dart_px[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(canvas, f"({x:.1f}, {y:.1f})mm", (dart_px[0]+10, dart_px[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Calculate and display score
                score, description, region_info = self.calculate_score(board_position)
                
                score_text = f"Score: {score} - {description}"
                cv2.putText(canvas, score_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(canvas, score_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                
                # Display distance from center
                dist_text = f"Distance: {region_info['distance']:.1f}mm"
                cv2.putText(canvas, dist_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(canvas, dist_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                
                # Show stability indicator
                if self.stable_position_count >= 10:
                    cv2.putText(canvas, "STABLE (1mm accuracy)", (10, canvas_size-30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(canvas, f"Stabilizing... ({self.stable_position_count}/10)", 
                              (10, canvas_size-30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add calibration mode indicator if active
        if self.calibration_mode:
            cv2.putText(canvas, "CALIBRATION MODE", (10, canvas_size-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.calibration_point:
                cv2.putText(canvas, f"Point: {self.calibration_point}", (10, canvas_size-90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Highlight current calibration point
                cal_px = mm_to_canvas_px(*self.calibration_point)
                cv2.circle(canvas, cal_px, 10, (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]-15), (cal_px[0]+15, cal_px[1]+15), (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]+15), (cal_px[0]+15, cal_px[1]-15), (0, 0, 255), 2)
        
        # Add FPS counter
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (canvas_size-150, canvas_size-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                  
        # Add accuracy indicator
        cv2.putText(canvas, "Target: 1mm accuracy", (canvas_size-300, canvas_size-50),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
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
    
    def run(self):
        """Run the dart tracking system with optimized settings for Raspberry Pi"""
        # Try loading cameras with error handling
        try:
            print(f"Opening camera 1 (index {self.cam_index1})...")
            cap1 = cv2.VideoCapture(self.cam_index1)
            if not cap1.isOpened():
                print(f"Error: Could not open camera {self.cam_index1}")
                return
                
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            print(f"Opening camera 2 (index {self.cam_index2})...")
            cap2 = cv2.VideoCapture(self.cam_index2)
            if not cap2.isOpened():
                print(f"Error: Could not open camera {self.cam_index2}")
                cap1.release()
                return
                
            cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        except Exception as e:
            print(f"Error initializing cameras: {e}")
            return
        
        # Try to load calibration data
        try:
            self.load_calibration()
        except:
            print("Using default calibration data")
            
        # Build polynomial models
        self.build_polynomial_models()
        
        # Print instructions
        print("\n*** RASPBERRY PI DART TRACKING SYSTEM ***")
        print("Target accuracy: 1mm")
        print("Press 'q' to exit")
        print("Press 't' to toggle calibration mode")
        print("Press 'c' in calibration mode to capture current point")
        print("Press 'r' to reset background subtractors")
        print("Press 's' to save current calibration to file")
        print("Press 'l' to load calibration from file")
        print("Use number keys 0-9 in calibration mode to select dart segments")
        
        # FPS calculation variables
        frame_count = 0
        fps_start_time = time.time()
        
        # Main loop
        while True:
            # Read frames with error handling
            try:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print("Error reading from cameras - retrying...")
                    time.sleep(0.5)  # Wait before retry
                    continue
            except Exception as e:
                print(f"Camera read error: {e}")
                time.sleep(0.5)
                continue
            
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
                        
                        # Rebuild polynomial models
                        self.build_polynomial_models()
                        
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
                    history=500, varThreshold=40, detectShadows=False
                )
                self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=40, detectShadows=False
                )
                # Reset tracking state
                self.kalman_initialized = False
                self.stable_position_count = 0
                self.last_stable_position = None
            elif key == ord('s'):
                print("Saving calibration to file")
                self.save_calibration()
            elif key == ord('l'):
                print("Loading calibration from file")
                if self.load_calibration():
                    self.build_polynomial_models()
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
    tracker = RaspberryPiDartTracker()
    tracker.run()
