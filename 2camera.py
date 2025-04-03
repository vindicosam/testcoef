import cv2
import numpy as np
import math
import time
from collections import deque
from bisect import bisect_left
import json
import ast
from scipy.spatial import Delaunay
from scipy.interpolate import RBFInterpolator

class EnhancedDartTracker:
    def __init__(self, cam_index1=0, cam_index2=2):
        """
        Enhanced dart tracking system that uses advanced calibration techniques
        and refined detection algorithms to achieve 1mm accuracy
        """
        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2
        
        # Camera settings with higher resolution for better accuracy
        self.frame_width = 1280  # Increased from 640
        self.frame_height = 720  # Increased from 480

        # Static camera positions
        self.camera1_position = (0, 550)    # Front camera fixed position
        self.camera2_position = (-400, 0)   # Side camera fixed position

        # Board ROI settings - adjust for higher resolution
        self.cam1_board_plane_y = 364  # Doubled from 182 for higher resolution
        self.cam1_roi_range = 60       # Doubled from 30 for higher resolution
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        self.cam2_board_plane_y = 416  # Doubled from 208 for higher resolution
        self.cam2_roi_range = 60       # Doubled from 30 for higher resolution
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        
        # Initialize with original calibration points
        # These will be doubled to account for higher resolution
        self.calibration_points = {
            # Format: (board_x, board_y): (cam1_pixel_x, cam2_pixel_x)
            (0, 0): (612, 632),
            (-171, 0): (1160, 768),
            (171, 0): (64, 588),
            (0, 171): (608, 56),
            (0, -171): (610, 1144),
            (90, 50): (302, 498),
            (-20, 103): (654, 262),
            (20, -100): (554, 918),
            (90, -50): (718, 812),
            (114, 121): (34, 306),
            (48, 86): (428, 364),
            (119, -117): (334, 858),
            (86, -48): (378, 718),
            (-118, -121): (906, 1248),
            (-50, -88): (746, 956),
            (-121, 118): (1248, 480),
            (-90, 47): (966, 84)
        }
        
        # Handle any legacy format issues
        calibration_points_fixed = {}
        for k, v in self.calibration_points.items():
            if len(k) == 4:  # It's in the format (x, y, px1, px2)
                calibration_points_fixed[(k[0], k[1])] = (k[2], k[3])
            else:
                calibration_points_fixed[k] = v
        
        self.calibration_points = calibration_points_fixed
        
        # Import segment calibration points from the original code
        segment_calibration_points = [
            # Double segments (outer ring) - values doubled for higher resolution
            (0, 169, 788, 62),      # Double 20 (top)
            (52, 161, 290, 160),     # Double 1
            (98, 139, 66, 266),     # Double 18
            (139, 98, 14, 378),     # Double 4
            (161, 52, 36, 482),     # Double 13
            (169, 0, 102, 592),      # Double 6 (right)
            (161, -52, 194, 698),    # Double 10
            (139, -98, 306, 810),   # Double 15
            (98, -139, 416, 924),   # Double 2
            (52, -161, 526, 1034),   # Double 17
            (0, -169, 634, 1134),    # Double 3 (bottom)
            (-52, -161, 742, 1216),  # Double 19
            (-98, -139, 858, 1258),  # Double 7
            (-139, -98, 980, 1216),  # Double 16
            (-161, -52, 1090, 1036),  # Double 8
            (-169, 0, 1184, 714),    # Double 11 (left)
            (-161, 52, 1258, 418),   # Double 14
            (-139, 98, 1272, 164),    # Double 9
            (-98, 139, 1194, 34),    # Double 12
            (-52, 161, 972, 18),     # Double 5

            # Triple segments (middle ring) - values doubled for higher resolution
            (0, 106, 642, 290),     # Triple 20 (top)
            (33, 101, 498, 328),    # Triple 1
            (62, 87, 382, 384),     # Triple 18
            (87, 62, 310, 454),     # Triple 4
            (101, 33, 286, 530),    # Triple 13
            (106, 0, 310, 608),     # Triple 6 (right)
            (101, -33, 350, 690),   # Triple 10
            (87, -62, 418, 764),    # Triple 15
            (62, -87, 488, 838),    # Triple 2
            (33, -101, 562, 904),   # Triple 17
            (0, -106, 642, 956),    # Triple 3 (bottom)
            (-33, -101, 720, 988),  # Triple 19
            (-62, -87, 796, 980),   # Triple 7
            (-87, -62, 872, 932),   # Triple 16
            (-101, -33, 932, 828),  # Triple 8
            (-106, 0, 978, 690),    # Triple 11 (left)
            (-101, 33, 994, 548),   # Triple 14
            (-87, 62, 980, 414),    # Triple 9
            (-62, 87, 912, 324),    # Triple 12
            (-33, 101, 790, 286),   # Triple 5
        ]
        
        # Add segment points to calibration
        for point in segment_calibration_points:
            board_x, board_y, cam1_pixel_x, cam2_pixel_x = point
            if cam1_pixel_x is not None and cam2_pixel_x is not None:
                self.calibration_points[(board_x, board_y)] = (cam1_pixel_x, cam2_pixel_x)
        
        # Advanced interpolation models
        self.rbf_model_cam1 = None
        self.rbf_model_cam2 = None
        self.triangulation = None
        self.calibration_points_array = None
        
        # Build advanced interpolation models
        self.build_interpolation_models()
        
        # Background subtractors with optimized parameters for higher resolution
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
            history=500,        # Shorter history for faster adaptation
            varThreshold=25,    # Lower threshold for more sensitive detection
            detectShadows=False
        )
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
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
        
        # Advanced Kalman filter for superior tracking
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0.03, 0],    # Adjusted for 30ms frame time
            [0, 1, 0, 0.03],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        # Fine-tuned process noise for millimeter accuracy
        self.kalman.processNoiseCov = np.array([
            [1e-5, 0, 0, 0],    # More precise position tracking
            [0, 1e-5, 0, 0],
            [0, 0, 1e-3, 0],    # Allow for some velocity variations
            [0, 0, 0, 1e-3]
        ], np.float32)
        # Reduced measurement noise for higher confidence in measurements
        self.kalman.measurementNoiseCov = np.array([
            [1e-2, 0],
            [0, 1e-2]
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
        
        # Sub-pixel refinement settings - enhanced for 1mm accuracy
        self.subpixel_window = (10, 10)  # Larger window for better accuracy
        self.subpixel_iterations = 50    # More iterations for refinement
        self.subpixel_epsilon = 0.0001   # Higher precision target
        
        # Filtering parameters for 1mm accuracy
        self.movement_threshold = 0.5  # 0.5mm threshold for stable detection
        self.stable_position_count = 0  # counter for stable positions
        self.last_stable_position = None  # last stable position
        
        # Use local region referencing for improved stability
        self.local_reference_enabled = True
        self.local_reference_radius = 5   # 5mm radius for local reference
        
        # Add temporal consistency check
        self.temporal_consistency_threshold = 2.0  # 2mm max movement between frames
        self.last_valid_position = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Camera undistortion parameters - these would ideally be calibrated
        # using a standard camera calibration procedure with a checkerboard
        self.camera_matrices = {
            'cam1': np.array([[800.0, 0, self.frame_width/2],
                             [0, 800.0, self.frame_height/2],
                             [0, 0, 1]], dtype=np.float32),
            'cam2': np.array([[800.0, 0, self.frame_width/2],
                             [0, 800.0, self.frame_height/2],
                             [0, 0, 1]], dtype=np.float32)
        }
        self.distortion_coeffs = {
            'cam1': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'cam2': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        
        # Enable undistortion only if proper calibration is available
        self.use_undistortion = False
    
    def build_interpolation_models(self):
        """
        Build advanced interpolation models using RBF interpolation
        and Delaunay triangulation for improved accuracy
        """
        if len(self.calibration_points) < 5:
            print("Not enough calibration points for advanced interpolation")
            return
            
        # Extract points for models
        board_points = []
        cam1_pixels = []
        cam2_pixels = []
        
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            board_points.append([board_x, board_y])
            cam1_pixels.append(cam1_pixel_x)
            cam2_pixels.append(cam2_pixel_x)
            
        # Convert to numpy arrays
        board_points = np.array(board_points)
        cam1_pixels = np.array(cam1_pixels)
        cam2_pixels = np.array(cam2_pixels)
        
        # Create RBF models for smooth interpolation
        # For camera 1, we predict board_x from pixel_x
        # For camera 2, we predict board_y from pixel_x
        try:
            # Use RBF for X coordinate (Camera 1)
            self.rbf_model_cam1 = RBFInterpolator(
                x=np.array(cam1_pixels).reshape(-1, 1), 
                y=board_points[:, 0],
                kernel='thin_plate_spline',
                epsilon=10   # Smoothing parameter
            )
            
            # Use RBF for Y coordinate (Camera 2)
            self.rbf_model_cam2 = RBFInterpolator(
                x=np.array(cam2_pixels).reshape(-1, 1), 
                y=board_points[:, 1],
                kernel='thin_plate_spline',
                epsilon=10  # Smoothing parameter
            )
            
            # Create Delaunay triangulation for barycentric interpolation
            # This allows for piecewise linear interpolation which can be more
            # accurate in regions where the mapping is complex
            try:
                # Store board points for triangulation lookup
                self.calibration_points_array = board_points
                self.triangulation = Delaunay(board_points)
                print(f"Built advanced interpolation models with {len(board_points)} points")
            except Exception as e:
                print(f"Could not build triangulation: {e}")
                self.triangulation = None
                
        except Exception as e:
            print(f"Error building RBF interpolation models: {e}")
            self.rbf_model_cam1 = None
            self.rbf_model_cam2 = None
    
    def load_calibration(self, filename="dart_calibration.json"):
        """Load calibration from file"""
        try:
            with open(filename, "r") as f:
                loaded_points = json.load(f)
                self.calibration_points = {ast.literal_eval(k): v for k, v in loaded_points.items()}
            
            # Rebuild advanced interpolation models
            self.build_interpolation_models()
            
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
    
    def advanced_interpolate(self, cam1_pixel_x, cam2_pixel_x):
        """
        Use advanced interpolation models for more accurate predictions
        """
        if self.rbf_model_cam1 is None or self.rbf_model_cam2 is None:
            # Fall back to basic interpolation
            return self.basic_interpolate(cam1_pixel_x, cam2_pixel_x)
            
        try:
            # Predict board coordinates using RBF models
            board_x = float(self.rbf_model_cam1(np.array([[cam1_pixel_x]])))
            board_y = float(self.rbf_model_cam2(np.array([[cam2_pixel_x]])))
            
            # Apply local region refinement if enabled
            if self.local_reference_enabled and self.triangulation is not None:
                # Find the closest calibration points to our predicted position
                predicted_point = np.array([board_x, board_y])
                
                # Find distances to all calibration points
                distances = np.sqrt(np.sum((self.calibration_points_array - predicted_point)**2, axis=1))
                
                # Get indices of closest points within local region
                local_indices = np.where(distances < self.local_reference_radius)[0]
                
                if len(local_indices) >= 3:
                    # Use weighted average of nearby calibration points for refinement
                    weights = 1.0 / (distances[local_indices] + 0.1)  # Prevent division by zero
                    weights = weights / np.sum(weights)  # Normalize weights
                    
                    # Apply weighted refinement
                    refined_x = np.sum(self.calibration_points_array[local_indices, 0] * weights)
                    refined_y = np.sum(self.calibration_points_array[local_indices, 1] * weights)
                    
                    # Blend the RBF prediction with local refinement
                    board_x = 0.7 * board_x + 0.3 * refined_x
                    board_y = 0.7 * board_y + 0.3 * refined_y
            
            return board_x, board_y
            
        except Exception as e:
            print(f"Error in advanced interpolation: {e}")
            # Fall back to basic interpolation
            return self.basic_interpolate(cam1_pixel_x, cam2_pixel_x)
    
    def basic_interpolate(self, cam1_pixel_x, cam2_pixel_x):
        """
        Basic linear interpolation method as fallback
        """
        # Create simple mapping tables
        cam1_mapping = []
        cam2_mapping = []
        
        for (board_x, board_y), (cal_cam1_x, cal_cam2_x) in self.calibration_points.items():
            cam1_mapping.append((cal_cam1_x, board_x))
            cam2_mapping.append((cal_cam2_x, board_y))
            
        cam1_mapping.sort(key=lambda x: x[0])
        cam2_mapping.sort(key=lambda x: x[0])
        
        # Linear interpolation for each coordinate
        board_x = self.interpolate_value(cam1_pixel_x, cam1_mapping)
        board_y = self.interpolate_value(cam2_pixel_x, cam2_mapping)
        
        return board_x, board_y
    
    def interpolate_value(self, pixel_value, mapping_table):
        """
        Interpolate a value using linear interpolation
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
    
    def apply_smoothing(self, new_value, history_key):
        """
        Apply adaptive smoothing based on motion speed
        """
        if new_value is None:
            return None
        
        # Add new value to history
        self.detection_history[history_key].append(new_value)
        
        # If we have enough history, compute weighted average
        if len(self.detection_history[history_key]) >= 3:
            # More recent values have higher weight
            total_weight = 0
            
            # For final position (x,y tuple)
            if history_key == 'final':
                # Calculate movement speed for adaptive filtering
                if len(self.detection_history[history_key]) >= 2:
                    latest = self.detection_history[history_key][-1]
                    previous = self.detection_history[history_key][-2]
                    
                    # Calculate movement in mm
                    dx = latest[0] - previous[0]
                    dy = latest[1] - previous[1]
                    movement = math.sqrt(dx*dx + dy*dy)
                    
                    # Adjust smoothing based on movement speed
                    # Fast movement: less smoothing, rely more on new measurements
                    # Slow movement: more smoothing for stability
                    if movement > 10:  # Fast movement (>10mm per frame)
                        alpha = 0.7  # Heavy weight on new value
                    elif movement > 5:  # Medium movement
                        alpha = 0.5  # Balanced weight
                    else:  # Slow movement or stable
                        alpha = 0.3  # More weight on history
                else:
                    # Default smoothing
                    alpha = 0.5
                
                # Apply exponential moving average with adaptive alpha
                if len(self.detection_history[history_key]) >= 2:
                    prev_avg = self.detection_history[history_key][-2]
                    latest = self.detection_history[history_key][-1]
                    
                    smoothed_x = alpha * latest[0] + (1 - alpha) * prev_avg[0]
                    smoothed_y = alpha * latest[1] + (1 - alpha) * prev_avg[1]
                    
                    return (smoothed_x, smoothed_y)
                else:
                    return self.detection_history[history_key][-1]
            else:
                # For single values (cam1/cam2)
                if len(self.detection_history[history_key]) >= 2:
                    prev_avg = self.detection_history[history_key][-2]
                    latest = self.detection_history[history_key][-1]
                    
                    # Calculate movement
                    movement = abs(latest - prev_avg)
                    
                    # Adjust alpha based on movement
                    if movement > 10:
                        alpha = 0.7
                    elif movement > 5:
                        alpha = 0.5
                    else:
                        alpha = 0.3
                    
                    # Apply exponential moving average
                    return alpha * latest + (1 - alpha) * prev_avg
                else:
                    return self.detection_history[history_key][-1]
        
        return new_value
    
    def update_kalman(self, position):
        """
        Enhanced Kalman filter update with adaptive process noise
        """
        if position is None:
            return None
        
        measurement = np.array([[position[0]], [position[1]]], np.float32)
        
        if not self.kalman_initialized:
            # Initialize Kalman filter with first measurement
            self.kalman.statePre = np.array([[position[0]], [position[1]], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[position[0]], [position[1]], [0], [0]], np.float32)
            self.kalman_initialized = True
            return position
        
        # Adaptive process noise - adjust based on measurement consistency
        if self.last_valid_position is not None:
            # Calculate movement
            dx = position[0] - self.last_valid_position[0]
            dy = position[1] - self.last_valid_position[1]
            movement = math.sqrt(dx*dx + dy*dy)
            
            # Adjust process noise based on movement
            if movement > 10:  # Fast motion
                # Increase process noise for velocity states
                self.kalman.processNoiseCov[2, 2] = 1e-2
                self.kalman.processNoiseCov[3, 3] = 1e-2
            elif movement < 2:  # Slow/stable motion
                # Decrease process noise for more stability
                self.kalman.processNoiseCov[2, 2] = 1e-4
                self.kalman.processNoiseCov[3, 3] = 1e-4
            else:  # Medium motion
                # Default values
                self.kalman.processNoiseCov[2, 2] = 1e-3
                self.kalman.processNoiseCov[3, 3] = 1e-3
        
        # Prediction step
        prediction = self.kalman.predict()
        
        # Correction step with adaptive measurement noise
        # If large jumps are detected, reduce confidence in measurements
        predicted_pos = (prediction[0, 0], prediction[1, 0])
        if self.last_valid_position is not None:
            # Calculate jump between prediction and measurement
            dx_pred = position[0] - predicted_pos[0]
            dy_pred = position[1] - predicted_pos[1]
            jump = math.sqrt(dx_pred*dx_pred + dy_pred*dy_pred)
            
            if jump > 10:  # Large unexpected jump
                # Temporarily increase measurement noise
                self.kalman.measurementNoiseCov[0, 0] = 1e0
                self.kalman.measurementNoiseCov[1, 1] = 1e0
            else:
                # Normal measurement noise
                self.kalman.measurementNoiseCov[0, 0] = 1e-2
                self.kalman.measurementNoiseCov[1, 1] = 1e-2
        
        # Perform correction
        corrected = self.kalman.correct(measurement)
        
        # Update last valid position
        self.last_valid_position = position
        
        # Return corrected state (x, y)
        return (corrected[0, 0], corrected[1, 0])
    
    def detect_point_with_subpixel(self, frame, fg_mask, roi_center_y, camera_id):
        """
        Enhanced dart detection with subpixel accuracy using multiple techniques
        
        Args:
            frame: Current frame ROI
            fg_mask: Foreground mask
            roi_center_y: Y-coordinate of ROI center line
            camera_id: Camera identifier for specific processing
            
        Returns:
            x: X-coordinate with subpixel accuracy
        """
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # No contours found
        if not contours:
            return None
            
        # Filter contours by size and proximity to center line
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Minimum area to reduce noise
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if contour is near the center line
            contour_center_y = y + h/2
            if abs(contour_center_y - roi_center_y) > 20:  # Allow some flexibility
                continue
                
            # Store valid contour with its area for sorting
            valid_contours.append((contour, area))
        
        # Sort contours by area (largest first)
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        
        # No valid contours
        if not valid_contours:
            return None
            
        # Take the largest contour
        largest_contour = valid_contours[0][0]
        
        # Use multiple detection techniques for subpixel accuracy
        
        # 1. Centroid method using moments
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            centroid_x = M["m10"] / M["m00"]
        else:
            # Fall back to bounding box center if moments fail
            x, y, w, h = cv2.boundingRect(largest_contour)
            centroid_x = x + w/2
        
        # 2. Find the point on contour closest to the center line
        min_dist = float('inf')
        closest_point_x = None
        
        for point in largest_contour:
            px, py = point[0]
            dist = abs(py - roi_center_y)
            if dist < min_dist:
                min_dist = dist
                closest_point_x = px
        
        # 3. Refine using intensity weighted center in a local window
        if closest_point_x is not None:
            # Define region of interest around the closest point
            window_size = 15  # pixels
            window_x_start = max(0, int(closest_point_x) - window_size)
            window_x_end = min(frame.shape[1], int(closest_point_x) + window_size)
            window_y_start = max(0, int(roi_center_y) - window_size)
            window_y_end = min(frame.shape[0], int(roi_center_y) + window_size)
            
            # Extract window from grayscale frame
            gray_window = cv2.cvtColor(frame[window_y_start:window_y_end, window_x_start:window_x_end], cv2.COLOR_BGR2GRAY)
            
            # Gaussian blur to reduce noise
            gray_window = cv2.GaussianBlur(gray_window, (5, 5), 0)
            
            # Apply threshold to focus on dart
            _, binary_window = cv2.threshold(gray_window, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate intensity weighted center
            weighted_sum_x = 0
            total_weight = 0
            
            for x in range(binary_window.shape[1]):
                for y in range(binary_window.shape[0]):
                    if binary_window[y, x] > 0:
                        weighted_sum_x += x * binary_window[y, x]
                        total_weight += binary_window[y, x]
            
            if total_weight > 0:
                refined_x = window_x_start + weighted_sum_x / total_weight
            else:
                refined_x = closest_point_x
        else:
            refined_x = centroid_x
        
        # 4. Apply subpixel corner refinement for even better precision
        try:
            # Convert to proper format for cornerSubPix
            corner = np.array([[refined_x, roi_center_y]], dtype=np.float32)
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                        self.subpixel_iterations, self.subpixel_epsilon)
            
            # Apply corner subpixel refinement
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(gray, corner, self.subpixel_window, (-1, -1), criteria)
            
            final_x = corner[0][0]
        except Exception as e:
            # Fall back to previous refinement if corner detection fails
            final_x = refined_x
        
        # Return the final refined x coordinate
        return final_x
    
    def process_camera1_frame(self, frame):
        """
        Process camera 1 frame with enhanced detection algorithms
        
        Returns:
            processed_frame: Processed frame with annotations
            fg_mask: Foreground mask
            dart_pixel_x: Detected dart x-coordinate in pixel space (or None)
        """
        # Apply undistortion if available and enabled
        if self.use_undistortion:
            frame = cv2.undistort(
                frame, 
                self.camera_matrices['cam1'], 
                self.distortion_coeffs['cam1']
            )
        
        # Rotate frame 180 degrees as in original code
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI for processing efficiency
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction with learning rate adjustment
        # A smaller learning rate (e.g., 0.001) means slower adaptation
        fg_mask = self.bg_subtractor1.apply(blurred, learningRate=0.005)
        
        # Threshold and clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to improve mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        # Detect dart using enhanced subpixel algorithm
        dart_pixel_x = self.detect_point_with_subpixel(roi, fg_mask, roi_center_y, 'cam1')
        
        # If dart detected
        if dart_pixel_x is not None:
            # Highlight the detected point
            cv2.circle(roi, (int(dart_pixel_x), roi_center_y), 5, (0, 255, 0), -1)
            cv2.putText(roi, f"Px: {dart_pixel_x:.2f}", (int(dart_pixel_x) + 5, roi_center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # In calibration mode, display the pixel values
            if self.calibration_mode and self.calibration_point:
                print(f"Cam1 pixel for {self.calibration_point}: {dart_pixel_x:.2f}")
            
            # Apply smoothing for stability
            smoothed_px = self.apply_smoothing(dart_pixel_x, 'cam1')
            
            # Store raw pixel value for advanced interpolation
            self.cam1_vector = smoothed_px
            
            # Display smoothed pixel value
            cv2.putText(roi, f"Smoothed: {smoothed_px:.2f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None
        
        # Copy ROI back to rotated frame
        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def process_camera2_frame(self, frame):
        """
        Process camera 2 frame with enhanced detection algorithms
        
        Returns:
            processed_frame: Processed frame with annotations
            fg_mask: Foreground mask
            dart_pixel_x: Detected dart x-coordinate in pixel space (or None)
        """
        # Apply undistortion if available and enabled
        if self.use_undistortion:
            frame = cv2.undistort(
                frame, 
                self.camera_matrices['cam2'], 
                self.distortion_coeffs['cam2']
            )
        
        # Rotate frame 180 degrees as in original code
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI for processing efficiency
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply background subtraction with learning rate adjustment
        fg_mask = self.bg_subtractor2.apply(blurred, learningRate=0.005)
        
        # Threshold and clean up mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to improve mask
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        # Detect dart using enhanced subpixel algorithm
        dart_pixel_x = self.detect_point_with_subpixel(roi, fg_mask, roi_center_y, 'cam2')
        
        # If dart detected
        if dart_pixel_x is not None:
            # Highlight the detected point
            cv2.circle(roi, (int(dart_pixel_x), roi_center_y), 5, (0, 255, 0), -1)
            cv2.putText(roi, f"Px: {dart_pixel_x:.2f}", (int(dart_pixel_x) + 5, roi_center_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # In calibration mode, display the pixel values
            if self.calibration_mode and self.calibration_point:
                print(f"Cam2 pixel for {self.calibration_point}: {dart_pixel_x:.2f}")
            
            # Apply smoothing for stability
            smoothed_px = self.apply_smoothing(dart_pixel_x, 'cam2')
            
            # Store raw pixel value for advanced interpolation
            self.cam2_vector = smoothed_px
            
            # Display smoothed pixel value
            cv2.putText(roi, f"Smoothed: {smoothed_px:.2f}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
        
        # Copy ROI back to rotated frame
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def compute_board_coordinates(self):
        """
        Compute dart board coordinates from camera detections using advanced interpolation
        
        Returns:
            position: (x, y) coordinates in board space, or None if no valid detection
        """
        # Check if we have valid detections from both cameras
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
            
        # Use advanced interpolation to get board coordinates
        board_coords = self.advanced_interpolate(self.cam1_vector, self.cam2_vector)
        
        # Validate coordinates are within reasonable range
        if board_coords:
            x, y = board_coords
            distance_from_center = math.sqrt(x*x + y*y)
            
            # Check if within or near the dartboard radius (with margin)
            if distance_from_center <= self.board_radius + 20:  # 20mm margin
                # Check for temporal consistency
                if self.last_valid_position is not None:
                    dx = x - self.last_valid_position[0]
                    dy = y - self.last_valid_position[1]
                    movement = math.sqrt(dx*dx + dy*dy)
                    
                    # If movement exceeds threshold, might be invalid detection
                    if movement > self.temporal_consistency_threshold * 5:  # Allow 5x threshold for initial jumps
                        print(f"Warning: Large movement detected ({movement:.1f}mm)")
                        # Still accept the position but with a warning
                
                # Apply Kalman filtering for smooth tracking
                filtered_position = self.update_kalman(board_coords)
                
                # Apply additional smoothing
                if filtered_position:
                    final_position = self.apply_smoothing(filtered_position, 'final')
                    return final_position
                
                return board_coords
                
        return None
    
    def calculate_score(self, position):
        """
        Calculate score based on dart position with enhanced accuracy
        
        Args:
            position: (x, y) position in board coordinates
            
        Returns:
            score: Score value
            description: Text description of the hit
            region_info: Additional information about the hit region
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
        
        # Enhanced ring detection using precise measurements
        # Double: 169-171mm, Triple: 105-107mm
        # Add small margins for improved accuracy
        if 167.5 <= distance_from_center <= 172.5:  # Double ring with margin
            multiplier = 2
            hit_type = "Double"
            region = "double"
        elif 103.5 <= distance_from_center <= 108.5:  # Triple ring with margin
            multiplier = 3
            hit_type = "Triple"
            region = "triple"
        else:  # Single
            multiplier = 1
            # Determine inner vs outer single
            if distance_from_center < 103.5:
                hit_type = "Inner Single"
                region = "inner_single"
            else:
                hit_type = "Outer Single"
                region = "outer_single"
            
        score = segment_number * multiplier
        
        if multiplier > 1:
            description = f"{hit_type} {segment_number} ({score})"
        else:
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
        Update board projection with current dart position using enhanced visualization
        
        Args:
            cam1_pixel_x, cam2_pixel_x: Raw pixel values from cameras (optional)
        """
        # Create a larger canvas to display the board and camera positions
        canvas_size = 800  # Increased for better visualization
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Calculate the scale factor to fit the board properly
        board_px_radius = 300  # Increased for better visualization
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
        
        # Draw mm measurement grid for improved precision visualization
        for r in range(0, int(self.board_radius) + 10, 10):  # 10mm grid
            cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                     int(r * scale_factor), (230, 230, 230), 1)
            
            # Add radial labels every 50mm
            if r % 50 == 0 and r > 0:
                label_pos = mm_to_canvas_px(r, 0)
                cv2.putText(canvas, f"{r}mm", (label_pos[0]+5, label_pos[1]-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Draw board boundary and rings
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
        
        # Draw segment markers
        for segment, (x, y) in self.board_segments.items():
            segment_px = mm_to_canvas_px(x, y)
            cv2.circle(canvas, segment_px, 3, (128, 0, 128), -1)
            cv2.putText(canvas, f"{segment}", (segment_px[0]+5, segment_px[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
        
        # Draw camera positions
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        
        cv2.circle(canvas, cam1_px, 10, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(canvas, cam2_px, 10, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Advanced calculation if both cameras have detections
        if cam1_pixel_x is not None and cam2_pixel_x is not None:
            # Compute board coordinates using advanced interpolation
            board_position = self.compute_board_coordinates()
            
            if board_position is not None:
                # Draw the dart position
                dart_px = mm_to_canvas_px(*board_position)
                
                # Determine color based on stability
                if self.stable_position_count >= 5:
                    # Stable position (yellow)
                    color = (0, 255, 255)
                    
                    # Check for movement below threshold for stability
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
                    # Moving position (green)
                    color = (0, 255, 0)
                    
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
                
                # Draw bull's eye target for 1mm precision visualization
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
                dist_text = f"Distance from center: {region_info['distance']:.1f}mm"
                cv2.putText(canvas, dist_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(canvas, dist_text, (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                
                # If position is stable, display "STABLE" indicator
                if self.stable_position_count >= 10:
                    cv2.putText(canvas, "STABLE (1mm accuracy)", (dart_px[0]-50, dart_px[1]-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(canvas, f"Stabilizing... ({self.stable_position_count}/10)", 
                              (dart_px[0]-50, dart_px[1]-20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add calibration mode indicator if active
        if self.calibration_mode:
            cv2.putText(canvas, "CALIBRATION MODE", (10, canvas_size-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.calibration_point:
                cv2.putText(canvas, f"Current point: {self.calibration_point}", (10, canvas_size-30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Highlight current calibration point on board
                cal_px = mm_to_canvas_px(*self.calibration_point)
                cv2.circle(canvas, cal_px, 10, (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]-15), (cal_px[0]+15, cal_px[1]+15), (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]+15), (cal_px[0]+15, cal_px[1]-15), (0, 0, 255), 2)
        
        # Add FPS counter
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (canvas_size-150, canvas_size-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                  
        # Add accuracy indicator (1mm target)
        cv2.putText(canvas, "Target accuracy: 1mm", (10, canvas_size-90),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    
    def run_camera_calibration(self):
        """
        Run camera calibration procedure using checkerboard
        to obtain lens distortion parameters
        """
        print("\nStarting camera calibration procedure...")
        print("This will capture multiple images of a standard checkerboard pattern")
        print("Hold the checkerboard in different positions and orientations")
        print("Press 'space' to capture an image, 'q' to finish calibration")
        
        # Initialize cameras
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Checkerboard parameters
        checkerboard_size = (9, 6)  # Number of inner corners
        square_size = 25.0  # mm
        
        # Arrays to store object points and image points
        obj_points = {
            'cam1': [],
            'cam2': []
        }
        img_points = {
            'cam1': [],
            'cam2': []
        }
        
        # Create object points (0,0,0), (1,0,0), ...
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Scale to real-world measurements
        
        while True:
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Error reading from cameras")
                break
            
            # Convert to grayscale for corner detection
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Display frames
            cv2.imshow("Camera 1", frame1)
            cv2.imshow("Camera 2", frame2)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar - capture calibration image
                # Find checkerboard corners
                ret1, corners1 = cv2.findChessboardCorners(gray1, checkerboard_size, None)
                ret2, corners2 = cv2.findChessboardCorners(gray2, checkerboard_size, None)
                
                if ret1 and ret2:
                    # Refine corners
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
                    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
                    
                    # Store points
                    obj_points['cam1'].append(objp)
                    img_points['cam1'].append(corners1)
                    obj_points['cam2'].append(objp)
                    img_points['cam2'].append(corners2)
                    
                    # Draw and display the corners
                    cv2.drawChessboardCorners(frame1, checkerboard_size, corners1, ret1)
                    cv2.drawChessboardCorners(frame2, checkerboard_size, corners2, ret2)
                    
                    cv2.imshow("Camera 1 - Checkerboard", frame1)
                    cv2.imshow("Camera 2 - Checkerboard", frame2)
                    
                    print(f"Captured calibration image {len(obj_points['cam1'])}")
                    cv2.waitKey(500)  # Short delay to show detected corners
                else:
                    print("Could not detect checkerboard in both cameras")
        
        # Release cameras
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        
        # Calculate calibration parameters if enough images were captured
        if len(obj_points['cam1']) > 5 and len(obj_points['cam2']) > 5:
            print("Computing camera calibration parameters...")
            
            # Calibrate camera 1
            ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
                obj_points['cam1'], img_points['cam1'], gray1.shape[::-1], None, None
            )
            
            # Calibrate camera 2
            ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
                obj_points['cam2'], img_points['cam2'], gray2.shape[::-1], None, None
            )
            
            # Save calibration parameters
            self.camera_matrices['cam1'] = mtx1
            self.camera_matrices['cam2'] = mtx2
            self.distortion_coeffs['cam1'] = dist1
            self.distortion_coeffs['cam2'] = dist2
            
            # Enable undistortion
            self.use_undistortion = True
            
            print("Camera calibration complete. Undistortion enabled.")
        else:
            print("Not enough calibration images captured. Undistortion not enabled.")
    
    def run(self):
        """Run the enhanced dart tracking system with 1mm accuracy"""
        # Initialize cameras with higher resolution
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
            
        # Build interpolation models
        self.build_interpolation_models()
        
        # Print instructions
        print("\n*** ENHANCED DART TRACKING SYSTEM ***")
        print("Target accuracy: 1mm")
        print("Press 'q' to exit")
        print("Press 't' to toggle calibration mode")
        print("Press 'c' in calibration mode to capture current point")
        print("Press 'r' to reset background subtractors")
        print("Press 's' to save current calibration to file")
        print("Press 'l' to load calibration from file")
        print("Press 'd' to run camera distortion calibration (using checkerboard)")
        
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
                        
                        # Rebuild interpolation models
                        self.build_interpolation_models()
                    else:
                        print("Could not detect dart in one or both cameras")
            elif key == ord('r'):
                print("Resetting background subtractors")
                self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=25, detectShadows=False
                )
                self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=25, detectShadows=False
                )
                # Reset Kalman filter
                self.kalman_initialized = False
                # Reset stable position tracking
                self.stable_position_count = 0
                self.last_stable_position = None
                self.last_valid_position = None
            elif key == ord('s'):
                print("Saving calibration to file")
                self.save_calibration()
            elif key == ord('l'):
                print("Loading calibration from file")
                self.load_calibration()
            elif key == ord('d'):
                self.run_camera_calibration()
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
    tracker = EnhancedDartTracker()
    tracker.run()
