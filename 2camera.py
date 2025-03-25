import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from queue import Queue
import threading
import subprocess
import signal
import sys
import cv2
import time
import os
import json
import matplotlib.image as mpimg
import math

class CameraStream:
    def __init__(self, device_index):
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.frame = None
        self.stopped = False
        self.device_index = device_index
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        if not self.cap.isOpened():
            print(f"Error: Could not open video device {self.device_index}")
            return None
        else:
            print(f"Video device {self.device_index} opened successfully")
            self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                print(f"Error: Failed to read frame from device {self.device_index}")

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
class LidarCameraVisualizer:
    def __init__(self):
        # Fixed LIDAR positions relative to the dartboard
        self.lidar1_pos = (-202.5, 224.0)  # Adjusted based on calibration
        self.lidar2_pos = (204.0, 223.5)   # Adjusted based on calibration

        # LIDAR configurations - refined based on calibration data
        self.lidar1_rotation = 342.5        # Adjusted angle
        self.lidar2_rotation = 186.25       # Adjusted angle
        self.lidar1_mirror = True
        self.lidar2_mirror = True
        self.lidar1_offset = 4.5            # Adjusted offset
        self.lidar2_offset = 0.5            # Adjusted offset

        # LIDAR heights above board surface - crucial for calculating true tip position
        self.lidar1_height = 4.0  # mm above board surface
        self.lidar2_height = 8.0  # mm above board surface

        # Camera 1 configuration (front camera)
        self.camera1_position = (0, 350)  # Camera is above the board
        self.camera1_vector_length = 1600  # Vector length in mm
        self.camera1_data = {"dart_mm_x": None, "dart_angle": None}

        # Camera 2 configuration (left side camera)
        self.camera2_position = (-350, 0)  # Camera is to the left of the board
        self.camera2_vector_length = 1600  # Vector length in mm
        self.camera2_data = {"dart_mm_y": None, "dart_angle": None}

        # Camera indices
        self.camera1_index = 0  # Front camera index
        self.camera2_index = 2  # Side camera index
        
        # ROI Settings for Camera 1 (front camera)
        self.camera1_board_plane_y = 198
        self.camera1_roi_range = 30
        self.camera1_roi_top = self.camera1_board_plane_y - self.camera1_roi_range
        self.camera1_roi_bottom = self.camera1_board_plane_y + self.camera1_roi_range
        self.camera1_pixel_to_mm_factor = -0.782  # Slope in mm/pixel
        self.camera1_pixel_offset = 226.8  # Board x when pixel_x = 0

        # ROI Settings for Camera 2 (side camera)
        self.camera2_board_plane_y = 199
        self.camera2_roi_range = 30
        self.camera2_roi_top = self.camera2_board_plane_y - self.camera2_roi_range
        self.camera2_roi_bottom = self.camera2_board_plane_y + self.camera2_roi_range
        self.camera2_pixel_to_mm_factor = -0.628  # Slope in mm/pixel
        self.camera2_pixel_offset = 192.8  # Board y when pixel_x = 0

        # Detection persistence
        self.last_valid_detection1 = {"dart_mm_x": None, "dart_angle": None}
        self.last_valid_detection2 = {"dart_mm_y": None, "dart_angle": None}
        self.detection_persistence_counter1 = 0
        self.detection_persistence_counter2 = 0
        self.detection_persistence_frames = 30

        # Camera background subtractors with sensitive parameters
        self.camera1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=75, varThreshold=15, detectShadows=False
        )
        self.camera2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=75, varThreshold=15, detectShadows=False
        )

        # LIDAR queues
        self.lidar1_queue = Queue()
        self.lidar2_queue = Queue()
        
        # Storage for most recent LIDAR data points
        self.lidar1_recent_points = []
        self.lidar2_recent_points = []
        self.max_recent_points = 20  # Keep last 20 points for smoothing

        # Store the projected LIDAR points (after lean compensation)
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None
        
        # Intersection points of camera vectors with board plane
        self.camera1_board_intersection = None
        self.camera2_board_intersection = None

        # Dartboard scaling
        self.board_scale_factor = 2.75
        self.dartboard_image = mpimg.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")

        # Radii for filtering points (including outer miss radius)
        self.radii = {
            "bullseye": 6.35,
            "outer_bull": 15.9,
            "inner_treble": 99,
            "outer_treble": 107,
            "inner_double": 162,
            "outer_double": 170,
            "board_edge": 195,  # Added outer radius for detecting misses
        }

        # --- Coefficient dictionaries (simplified for brevity) ---
        self.large_segment_coeff = {}  # Coefficients for the outer single area
        self.doubles_coeff = {}        # Coefficients for the double ring area
        self.trebles_coeff = {}        # Coefficients for the triple ring area
        self.small_segment_coeff = {}  # Coefficients for the inner single area

        # Running flag
        self.running = True

        # Calibration correction matrix based on provided screenshots
        self.calibration_points = {
            (0, 0): (-1.0, 0),  # Bullseye - significant offset correction
            (23, 167): (0.9, 2.7),  # Singles area (outer)
            (-23, -167): (2.4, -2.6),  # Singles area (outer)
            (167, -24): (2.9, 2.6),  # Singles area (outer)
            (-167, 24): (-3.3, -2.7),  # Singles area (outer)
            (75, 75): (2.3, 2.1),  # Trebles area
            (-75, -75): (2.2, -3.2),  # Trebles area - corrected point
        }
        
        self.x_scale_correction = 1.02  # Slight adjustment for X scale
        self.y_scale_correction = 1.04  # Slight adjustment for Y scale
        
        # 3D lean detection variables
        self.current_forward_lean_angle = 0.0
        self.current_side_lean_angle = 0.0
        self.forward_lean_confidence = 0.0
        self.side_lean_confidence = 0.0
        self.lean_history_forward = []  # Store recent forward lean readings for smoothing
        self.lean_history_side = []  # Store recent side lean readings for smoothing
        self.max_lean_history = 60  # Keep track of last 60 lean readings
        self.forward_lean_arrow = None  # For visualization
        self.side_lean_arrow = None  # For visualization
        self.forward_arrow_text = None  # For visualization text
        self.side_arrow_text = None  # For visualization text
        
        # Maximum expected lean angles
        self.MAX_SIDE_LEAN = 35.0  # Maximum expected side-to-side lean in degrees
        self.MAX_FORWARD_LEAN = 30.0  # Maximum expected forward/backward lean in degrees
        
        # Maximum expected Y-difference for maximum lean (calibration parameter)
        self.MAX_Y_DIFF_FOR_MAX_LEAN = 9.0  # mm
        self.MAX_X_DIFF_FOR_MAX_LEAN = 9.0  # mm

        # Calibration factors for lean correction - limited to 8mm as requested
        self.side_lean_max_adjustment = 8.0  # mm, maximum adjustment for side lean
        self.forward_lean_max_adjustment = 8.0  # mm, maximum adjustment for forward lean
        
        # Coefficient strength scaling factors (per segment and ring)
        self.coefficient_scaling = {}
        
        # Set default values for all segments (1-20)
        for segment in range(1, 21):
            self.coefficient_scaling[segment] = {
                'doubles': 1.0,  # Scale for double ring
                'trebles': 1.0,  # Scale for treble ring
                'small': 1.0,    # Scale for inner single area (small segments)
                'large': 1.0     # Scale for outer single area (large segments)
            }
        
        # Setup visualization
        self.setup_plot()

        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
 def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        if hasattr(self, 'camera1_stream') and self.camera1_stream:
            self.camera1_stream.stop()
        if hasattr(self, 'camera2_stream') and self.camera2_stream:
            self.camera2_stream.stop()
        plt.close("all")
        sys.exit(0)

    def setup_plot(self):
        """Initialize the plot with enhanced visualization for 3D lean."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_title("LIDAR and Dual Camera Vector Visualization")
        self.ax.grid(True)

        # Add dartboard image
        self.update_dartboard_image()

        # Plot LIDAR and camera positions
        self.ax.plot(*self.lidar1_pos, "bo", label="LIDAR 1")
        self.ax.plot(*self.lidar2_pos, "go", label="LIDAR 2")
        self.ax.plot(*self.camera1_position, "ro", label="Camera 1 (Front)")
        self.ax.plot(*self.camera2_position, "mo", label="Camera 2 (Left)")

        # Draw all radii circles
        for name, radius in self.radii.items():
            circle = plt.Circle(
                (0, 0), radius, fill=False, linestyle="--", color="gray", alpha=0.4
            )
            self.ax.add_patch(circle)
            self.ax.text(
                0, radius, name, color="gray", fontsize=8, ha="center", va="bottom"
            )

        # Vectors and detected dart position
        (self.scatter1,) = self.ax.plot([], [], "b.", label="LIDAR 1 Data")
        (self.scatter2,) = self.ax.plot([], [], "g.", label="LIDAR 2 Data")
        (self.camera1_vector,) = self.ax.plot([], [], "r--", label="Camera 1 Vector")
        (self.camera2_vector,) = self.ax.plot([], [], "m--", label="Camera 2 Vector")
        (self.lidar1_vector,) = self.ax.plot([], [], "b--", label="LIDAR 1 Vector")
        (self.lidar2_vector,) = self.ax.plot([], [], "g--", label="LIDAR 2 Vector")
        (self.camera1_dart,) = self.ax.plot(
            [], [], "rx", markersize=8, label="Camera 1 Intersection"
        )
        (self.camera2_dart,) = self.ax.plot(
            [], [], "mx", markersize=8, label="Camera 2 Intersection"
        )
        (self.lidar1_dart,) = self.ax.plot(
            [], [], "bx", markersize=8, label="LIDAR 1 Projected"
        )
        (self.lidar2_dart,) = self.ax.plot(
            [], [], "gx", markersize=8, label="LIDAR 2 Projected"
        )
        (self.detected_dart,) = self.ax.plot(
            [], [], "ko", markersize=10, label="Final Tip Position"
        )

        # Add text annotation for lean angles
        self.lean_text = self.ax.text(-380, 380, "", fontsize=9)

        self.ax.legend(loc="upper right", fontsize=8)

    def update_dartboard_image(self):
        """Update the dartboard image extent."""
        scaled_extent = [
            -170 * self.board_scale_factor,
            170 * self.board_scale_factor,
            -170 * self.board_scale_factor,
            170 * self.board_scale_factor,
        ]
        self.ax.imshow(self.dartboard_image, extent=scaled_extent, zorder=0)

    def calculate_dart_angle(self, contour, mask):
        """
        Calculate the angle of the dart tip relative to vertical using linear regression.
        Returns angle in degrees where:
        - 90 degrees = perfectly upright (perpendicular to board)
        - 0 degrees = flat against the board (parallel)
        """
        if len(contour) < 5:
            return None, None

        # Find the tip point (highest point in the contour)
        tip_point = None
        for point in contour:
            x, y = point[0]
            if tip_point is None or y < tip_point[1]:
                tip_point = (x, y)
        
        if tip_point is None:
            return None, None
            
        tip_x, tip_y = tip_point
        
        # Look for points in the mask below the tip
        points_below = []
        search_depth = 20  # How far down to look
        search_width = 40  # How wide to search
        
        # Define region to search
        min_x = max(0, tip_x - search_width)
        max_x = min(mask.shape[1] - 1, tip_x + search_width)
        max_y = min(mask.shape[0] - 1, tip_y + search_depth)
        
        # Find all white pixels in the search area
        for y in range(tip_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if mask[y, x] > 0:  # White pixel
                    points_below.append((x, y))
        
        if len(points_below) < 5:  # Need enough points for a good fit
            return None, None
            
        # Use linear regression to find the best fit line through the points
        points = np.array(points_below)
        x = points[:, 0]
        y = points[:, 1]
        
        # Calculate slope using least squares
        n = len(points)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate slope
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:  # Avoid division by zero
            slope = float('inf')
            angle = 90  # Vertical line
        else:
            slope = numerator / denominator
            # Convert slope to angle in degrees (relative to vertical, not horizontal)
            angle_from_horizontal = np.degrees(np.arctan(slope))
            angle = 90 - angle_from_horizontal  # Convert to angle from vertical
        
        # Determine lean direction
        lean = "VERTICAL"
        if angle < 85:
            lean = "LEFT"
        elif angle > 95:
            lean = "RIGHT"
            
        return angle, (lean, points_below)

    def camera1_detection(self):
        """Detect dart tip using the front camera (Camera 1)."""
        self.camera1_stream = CameraStream(self.camera1_index).start()
        if self.camera1_stream is None:
            print("Error: Failed to start Camera 1")
            return
        
        while self.running:
            frame = self.camera1_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Flip the frame 180 degrees since camera is upside down
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            roi = frame[self.camera1_roi_top : self.camera1_roi_bottom, :]

            # Background subtraction and thresholding
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            fg_mask = self.camera1_bg_subtractor.apply(gray)

            # Threshold
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

            # Morphological operations to enhance the dart
            kernel = np.ones((3, 3), np.uint8)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

            # Reset current detection
            self.camera1_data["dart_mm_x"] = None
            self.camera1_data["dart_angle"] = None

            # Detect contours
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Find the dart tip (highest point since image is flipped)
                tip_contour = None
                tip_point = None
                
                for contour in contours:
                    if cv2.contourArea(contour) > 20:
                        x, y, w, h = cv2.boundingRect(contour)
                        dart_pixel_x = x + w // 2
                        
                        # Use the board plane as the y-position
                        roi_center_y = self.camera1_board_plane_y - self.camera1_roi_top
                        
                        if tip_contour is None:
                            tip_contour = contour
                            tip_point = (dart_pixel_x, roi_center_y)

                if tip_contour is not None and tip_point is not None:
                    # Calculate dart angle
                    dart_angle, additional_info = self.calculate_dart_angle(tip_contour, fg_mask)

                    # Map pixels to mm coordinates using epipolar calibration
                    dart_mm_x = self.camera1_pixel_to_mm_factor * tip_point[0] + self.camera1_pixel_offset

                    # Save data
                    self.camera1_data["dart_mm_x"] = dart_mm_x
                    self.camera1_data["dart_angle"] = dart_angle

                    # Update persistence
                    self.last_valid_detection1 = self.camera1_data.copy()
                    self.detection_persistence_counter1 = self.detection_persistence_frames

            # If no dart detected but we have a valid previous detection
            elif self.detection_persistence_counter1 > 0:
                self.detection_persistence_counter1 -= 1
                if self.detection_persistence_counter1 > 0:
                    self.camera1_data = self.last_valid_detection1.copy()

        if self.camera1_stream:
            self.camera1_stream.stop()

    def camera2_detection(self):
        """Detect dart tip using the left side camera (Camera 2)."""
        self.camera2_stream = CameraStream(self.camera2_index).start()
        if self.camera2_stream is None:
            print("Error: Failed to start Camera 2")
            return
            
        while self.running:
            frame = self.camera2_stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Flip the frame 180 degrees to match front camera orientation
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            roi = frame[self.camera2_roi_top : self.camera2_roi_bottom, :]

            # Background subtraction and thresholding
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            fg_mask = self.camera2_bg_subtractor.apply(gray)

            # Threshold
            fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

            # Morphological operations to enhance the dart
            kernel = np.ones((3, 3), np.uint8)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

            # Reset current detection
            self.camera2_data["dart_mm_y"] = None
            self.camera2_data["dart_angle"] = None

            # Detect contours
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Find the dart tip
                tip_contour = None
                tip_point = None
                
                for contour in contours:
                    if cv2.contourArea(contour) > 20:
                        x, y, w, h = cv2.boundingRect(contour)
                        dart_pixel_x = x + w // 2
                        
                        # Use the board plane as the y-position
                        roi_center_y = self.camera2_board_plane_y - self.camera2_roi_top
                        
                        if tip_contour is None:
                            tip_contour = contour
                            tip_point = (dart_pixel_x, roi_center_y)

                if tip_contour is not None and tip_point is not None:
                    # Calculate dart angle
                    dart_angle, additional_info = self.calculate_dart_angle(tip_contour, fg_mask)

                    # Map pixels to mm coordinates using epipolar calibration
                    dart_mm_y = self.camera2_pixel_to_mm_factor * tip_point[0] + self.camera2_pixel_offset

                    # Save data
                    self.camera2_data["dart_mm_y"] = dart_mm_y
                    self.camera2_data["dart_angle"] = dart_angle

                    # Update persistence
                    self.last_valid_detection2 = self.camera2_data.copy()
                    self.detection_persistence_counter2 = self.detection_persistence_frames

            # If no dart detected but we have a valid previous detection
            elif self.detection_persistence_counter2 > 0:
                self.detection_persistence_counter2 -= 1
                if self.detection_persistence_counter2 > 0:
                    self.camera2_data = self.last_valid_detection2.copy()

        if self.camera2_stream:
            self.camera2_stream.stop()
def detect_side_to_side_lean(self, lidar1_point, lidar2_point):
        """
        Detect side-to-side lean using camera data as primary source when available,
        and LIDAR data as secondary.
        
        Returns:
            lean_angle: Estimated side lean angle in degrees (0° = vertical, positive = toward right, negative = toward left)
            confidence: Confidence level in the lean detection (0-1)
        """
        # First, check if we have camera angle data from top camera (higher priority)
        camera1_angle = self.camera1_data.get("dart_angle")
        
        if camera1_angle is not None:
            # Convert camera angle to lean angle (90° is vertical)
            # Positive lean = leaning right, Negative lean = leaning left
            side_lean_angle = 90 - camera1_angle
            
            # Higher confidence for direct camera measurement
            confidence = 0.9
            return side_lean_angle, confidence
            
        # Fall back to LIDAR-based detection if no camera data
        if lidar1_point is None or lidar2_point is None:
            return 0, 0  # No lean detected, zero confidence

        # Extract x-coordinates from both LIDARs
        x1 = lidar1_point[0]
        x2 = lidar2_point[0]

        # Calculate x-difference as indicator of side-to-side lean
        # If perfectly vertical, x1 and x2 should be very close
        x_diff = x1 - x2

        # Convert x-difference to an angle
        MAX_LEAN_ANGLE = self.MAX_SIDE_LEAN
        MAX_X_DIFF = self.MAX_X_DIFF_FOR_MAX_LEAN  # mm

        # Calculate angle as proportion of max difference
        lean_angle = (x_diff / MAX_X_DIFF) * MAX_LEAN_ANGLE

        # Clamp to reasonable range
        lean_angle = max(-MAX_LEAN_ANGLE, min(MAX_LEAN_ANGLE, lean_angle))

        # Calculate confidence
        confidence = min(
            0.7,  # Lower max confidence for LIDAR-only detection
            (len(self.lidar1_recent_points) + len(self.lidar2_recent_points))
            / (2 * self.max_recent_points),
        )

        return lean_angle, confidence

    def detect_forward_backward_lean(self, lidar1_point, lidar2_point):
        """
        Detect forward/backward lean using side camera data as primary source when available,
        and LIDAR data as secondary.
        
        Returns:
            lean_angle: Estimated forward lean angle in degrees (0° = vertical, positive = toward front, negative = toward back)
            confidence: Confidence level in the lean detection (0-1)
        """
        # First, check if we have camera angle data from side camera (higher priority)
        camera2_angle = self.camera2_data.get("dart_angle")
        
        if camera2_angle is not None:
            # Convert camera angle to lean angle (90° is vertical)
            # Positive lean = leaning forward, Negative lean = leaning backward
            forward_lean_angle = 90 - camera2_angle
            
            # Higher confidence for direct camera measurement
            confidence = 0.9
            return forward_lean_angle, confidence
            
        # Fall back to LIDAR-based detection if no camera data
        if lidar1_point is None or lidar2_point is None:
            return 0, 0  # No lean detected, zero confidence

        # Extract y-coordinates from both LIDARs
        y1 = lidar1_point[1]
        y2 = lidar2_point[1]

        # Calculate y-difference as indicator of forward/backward lean
        # If perfectly vertical, y1 and y2 should be very close
        y_diff = y1 - y2

        # Convert y-difference to an angle
        MAX_LEAN_ANGLE = self.MAX_FORWARD_LEAN
        MAX_Y_DIFF = self.MAX_Y_DIFF_FOR_MAX_LEAN  # mm

        # Calculate angle as proportion of max difference
        lean_angle = (y_diff / MAX_Y_DIFF) * MAX_LEAN_ANGLE

        # Clamp to reasonable range
        lean_angle = max(-MAX_LEAN_ANGLE, min(MAX_LEAN_ANGLE, lean_angle))

        # Calculate confidence
        confidence = min(
            0.7,  # Lower max confidence for LIDAR-only detection
            (len(self.lidar1_recent_points) + len(self.lidar2_recent_points))
            / (2 * self.max_recent_points),
        )

        return lean_angle, confidence
def project_lidar_point_with_3d_lean(
        self,
        lidar_point,
        lidar_height,
        forward_lean_angle,
        side_lean_angle,
        camera1_x,
        camera2_y,
    ):
        """
        Project a LIDAR detection point to account for 3D lean using data from two cameras.
        Maximum correction is limited to 8mm as requested.
        
        Corrections are applied in the OPPOSITE direction of the lean:
        - If leaning right (positive side_lean_angle), correct LEFT
        - If leaning left (negative side_lean_angle), correct RIGHT
        - If leaning forward (positive forward_lean_angle), correct BACKWARD
        - If leaning backward (negative forward_lean_angle), correct FORWARD
        """
        if lidar_point is None:
            return lidar_point

        # Handle missing lean angles
        if forward_lean_angle is None:
            forward_lean_angle = 0  # Default to vertical
        if side_lean_angle is None:
            side_lean_angle = 0  # Default to vertical

        # Extract original coordinates
        original_x, original_y = lidar_point

        # Apply side-to-side lean adjustment (opposite to the lean direction)
        adjusted_x = original_x
        if side_lean_angle != 0:
            # Calculate side-to-side lean adjustment factor (0-1)
            side_lean_factor = abs(side_lean_angle) / self.MAX_SIDE_LEAN
            
            # Calculate maximum adjustment magnitude
            max_adjustment = self.side_lean_max_adjustment * side_lean_factor
            
            # Direction is OPPOSITE to lean direction:
            # Positive side_lean = leaning right, so adjust LEFT (negative)
            # Negative side_lean = leaning left, so adjust RIGHT (positive)
            side_adjustment = -max_adjustment if side_lean_angle > 0 else max_adjustment
            
            # Apply side-to-side adjustment
            adjusted_x = original_x + side_adjustment
            
            # Debug output
            print(f"Side lean: {side_lean_angle:.1f}° → Adjusting X by {side_adjustment:.2f}mm")

        # Apply forward/backward lean adjustment (opposite to the lean direction)
        adjusted_y = original_y
        if forward_lean_angle != 0:
            # Calculate forward/backward lean adjustment factor (0-1)
            forward_lean_factor = abs(forward_lean_angle) / self.MAX_FORWARD_LEAN
            
            # Calculate maximum adjustment magnitude
            max_adjustment = self.forward_lean_max_adjustment * forward_lean_factor
            
            # Direction is OPPOSITE to lean direction:
            # Positive forward_lean = leaning forward, so adjust BACKWARD (negative) 
            # Negative forward_lean = leaning backward, so adjust FORWARD (positive)
            forward_adjustment = -max_adjustment if forward_lean_angle > 0 else max_adjustment
            
            # Apply forward/backward adjustment
            adjusted_y = original_y + forward_adjustment
            
            # Debug output
            print(f"Forward lean: {forward_lean_angle:.1f}° → Adjusting Y by {forward_adjustment:.2f}mm")

        # The result is now corrected in opposition to the lean
        return (adjusted_x, adjusted_y)
def find_camera1_board_intersection(self, camera_x):
        """Calculate where the front camera epipolar line intersects the board surface."""
        if camera_x is None:
            return None

        # Using epipolar geometry, the X coordinate is from camera, Y is 0 (board surface)
        return (camera_x, 0)

    def find_camera2_board_intersection(self, camera_y):
        """Calculate where the side camera epipolar line intersects the board surface."""
        if camera_y is None:
            return None

        # Using epipolar geometry, the Y coordinate is from camera, X is 0 (board center)
        return (0, camera_y)

    def compute_epipolar_intersection(self):
        """
        Compute the intersection of the vectors from both cameras using epipolar geometry.
        This finds the 3D position of the dart tip.
        """
        if self.camera1_data.get("dart_mm_x") is None or self.camera2_data.get("dart_mm_y") is None:
            return None
            
        # For cam1, we've determined the board_x value where the vector passes through the board plane
        cam1_board_x = self.camera1_data.get("dart_mm_x")
        cam1_ray_start = self.camera1_position
        cam1_ray_end = (cam1_board_x, 0)  # This is where the ray passes through the board
        
        # For cam2, we've determined the board_y value where the vector passes through the board plane
        cam2_board_y = self.camera2_data.get("dart_mm_y")
        cam2_ray_start = self.camera2_position
        cam2_ray_end = (0, cam2_board_y)  # This is where the ray passes through the board
        
        # Find the intersection of these rays
        intersection = self.compute_line_intersection(
            cam1_ray_start, cam1_ray_end, 
            cam2_ray_start, cam2_ray_end
        )
        
        return intersection

    def calculate_final_tip_position(
        self, camera1_point, camera2_point, lidar1_point, lidar2_point
    ):
        """
        Calculate the final tip position using all available data, prioritizing camera data.

        Args:
            camera1_point: Intersection of front camera vector with board
            camera2_point: Intersection of side camera vector with board
            lidar1_point: Projected LIDAR 1 point
            lidar2_point: Projected LIDAR 2 point

        Returns:
            (x, y) final estimated tip position
        """
        # First, try to compute epipolar intersection if both cameras have data
        epipolar_point = self.compute_epipolar_intersection()
        if epipolar_point is not None:
            # We have a direct intersection from both cameras - highest priority
            return epipolar_point
        
        # If no epipolar intersection, continue with traditional fusion approach
        # Points that are actually available
        valid_points = []
        if camera1_point is not None:
            valid_points.append(camera1_point)
        if camera2_point is not None:
            valid_points.append(camera2_point)
        if lidar1_point is not None:
            valid_points.append(lidar1_point)
        if lidar2_point is not None:
            valid_points.append(lidar2_point)

        # If no valid points, can't determine position
        if not valid_points:
            return None

        # If only one sensor has data, use that
        if len(valid_points) == 1:
            return valid_points[0]

        # First priority: Both cameras have data - use their intersection
        if camera1_point is not None and camera2_point is not None:
            # Take X from front camera and Y from side camera (more accurate)
            camera_fusion_x = camera1_point[0]
            camera_fusion_y = camera2_point[1]
            camera_fusion_point = (camera_fusion_x, camera_fusion_y)

            # If we also have LIDAR data, incorporate it with lower weight
            if lidar1_point is not None and lidar2_point is not None:
                # Camera data gets higher weight 
                camera_weight = 0.8  # Increased weight for camera data
                lidar_weight = 0.2  # Reduced weight for LIDAR data

                # Calculate weighted average
                final_x = (
                    camera_fusion_point[0] * camera_weight
                    + ((lidar1_point[0] + lidar2_point[0]) / 2) * lidar_weight
                )
                final_y = (
                    camera_fusion_point[1] * camera_weight
                    + ((lidar1_point[1] + lidar2_point[1]) / 2) * lidar_weight
                )

                return (final_x, final_y)

            # If we only have camera data (no LIDAR)
            return camera_fusion_point

        # Second priority: Front camera + LIDAR
        elif camera1_point is not None:
            if lidar1_point is not None and lidar2_point is not None:
                # Use X from camera (more accurate) and Y from average of LIDARs
                return (camera1_point[0], (lidar1_point[1] + lidar2_point[1]) / 2)
            elif lidar1_point is not None:
                return (camera1_point[0], lidar1_point[1])
            elif lidar2_point is not None:
                return (camera1_point[0], lidar2_point[1])
            else:
                return camera1_point  # Only camera1 data available

        # Third priority: Side camera + LIDAR
        elif camera2_point is not None:
            if lidar1_point is not None and lidar2_point is not None:
                # Use Y from camera (more accurate) and X from average of LIDARs
                return ((lidar1_point[0] + lidar2_point[0]) / 2, camera2_point[1])
            elif lidar1_point is not None:
                return (lidar1_point[0], camera2_point[1])
            elif lidar2_point is not None:
                return (lidar2_point[0], camera2_point[1])
            else:
                return camera2_point  # Only camera2 data available

        # Fourth priority: Only LIDAR data (no cameras)
        if lidar1_point is not None and lidar2_point is not None:
            # Simple average if no camera data
            final_x = (lidar1_point[0] + lidar2_point[0]) / 2
            final_y = (lidar1_point[1] + lidar2_point[1]) / 2
            return (final_x, final_y)
        elif lidar1_point is not None:
            return lidar1_point
        elif lidar2_point is not None:
            return lidar2_point

        # This shouldn't happen if the earlier logic is correct
        return valid_points[0]
def run(self, lidar1_script, lidar2_script):
        """Start all components."""
        lidar1_thread = threading.Thread(target=self.start_lidar, args=(lidar1_script, self.lidar1_queue, 1))
        lidar2_thread = threading.Thread(target=self.start_lidar, args=(lidar2_script, self.lidar2_queue, 2))
        
        # Make all threads daemon threads
        lidar1_thread.daemon = True
        lidar2_thread.daemon = True
        
        print("Starting LIDAR 1...")
        lidar1_thread.start()
        time.sleep(2)
        print("Starting LIDAR 2...")
        lidar2_thread.start()
        time.sleep(2)
        
        # Start camera threads last
        print("Starting cameras...")
        camera1_thread = threading.Thread(target=self.camera1_detection)
        camera2_thread = threading.Thread(target=self.camera2_detection)
        camera1_thread.daemon = True
        camera2_thread.daemon = True
        camera1_thread.start()
        time.sleep(1)
        camera2_thread.start()

        print("All sensors started. Beginning visualization...")
        anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)
        plt.show()

        # Cleanup
        self.running = False
        print("Shutting down threads...")

if __name__ == "__main__":
    lidar1_script = "./tri_test_lidar1"
    lidar2_script = "./tri_test_lidar2"
    visualizer = LidarCameraVisualizer()
    
    # Try to load coefficient scaling from file
    visualizer.load_coefficient_scaling()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            print("Calibration Mode")
            print("1. LIDAR Rotation Calibration")
            print("2. Coefficient Scaling Calibration")
            print("q. Quit")
            
            option = input("Select option: ")
            
            if option == "1":
                print("LIDAR Rotation Calibration Mode")
                print(f"Current LIDAR1 rotation: {visualizer.lidar1_rotation}°")
                print(f"Current LIDAR2 rotation: {visualizer.lidar2_rotation}°")
                
                while True:
                    cmd = input("Enter L1+/L1-/L2+/L2- followed by degrees (e.g., L1+0.5) or 'q' to quit: ")
                    if cmd.lower() == 'q':
                        break
                        
                    try:
                        if cmd.startswith("L1+"):
                            visualizer.lidar1_rotation += float(cmd[3:])
                        elif cmd.startswith("L1-"):
                            visualizer.lidar1_rotation -= float(cmd[3:])
                        elif cmd.startswith("L2+"):
                            visualizer.lidar2_rotation += float(cmd[3:])
                        elif cmd.startswith("L2-"):
                            visualizer.lidar2_rotation -= float(cmd[3:])
                            
                        print(f"Updated LIDAR1 rotation: {visualizer.lidar1_rotation}°")
                        print(f"Updated LIDAR2 rotation: {visualizer.lidar2_rotation}°")
                    except:
                        print("Invalid command format")
            elif option == "2":
                print("Coefficient Scaling Calibration Mode")
                print("Adjust scaling factors for specific segments and ring types.")
                print("Format: [segment]:[ring_type]:[scale]")
                print("  - segment: 1-20 or 'all'")
                print("  - ring_type: 'doubles', 'trebles', 'small', 'large', or 'all'")
                print("  - scale: scaling factor (e.g. 0.5, 1.0, 1.5)")
                
                while True:
                    cmd = input("Enter scaling command or 'q' to quit: ")
                    if cmd.lower() == 'q':
                        break
                        
                    try:
                        parts = cmd.split(':')
                        if len(parts) != 3:
                            print("Invalid format. Use segment:ring_type:scale")
                            continue
                            
                        segment_str, ring_type, scale_str = parts
                        scale = float(scale_str)
                        
                        # Process segment specification
                        segments = []
                        if segment_str.lower() == 'all':
                            segments = list(range(1, 21))
                        else:
                            try:
                                segment_num = int(segment_str)
                                if 1 <= segment_num <= 20:
                                    segments = [segment_num]
                                else:
                                    print("Segment must be between 1-20 or 'all'")
                                    continue
                            except ValueError:
                                print("Segment must be a number between 1-20 or 'all'")
                                continue
                        
                        # Process ring type specification
                        ring_types = []
                        if ring_type.lower() == 'all':
                            ring_types = ['doubles', 'trebles', 'small', 'large']
                        elif ring_type.lower() in ['doubles', 'trebles', 'small', 'large']:
                            ring_types = [ring_type.lower()]
                        else:
                            print("Ring type must be 'doubles', 'trebles', 'small', 'large', or 'all'")
                            continue
                        
                        # Update scaling factors
                        for segment in segments:
                            for rt in ring_types:
                                visualizer.coefficient_scaling[segment][rt] = scale
                                
                        print(f"Updated scaling factors for {len(segments)} segment(s) and {len(ring_types)} ring type(s)")
                    except ValueError:
                        print("Scale must be a numeric value")
            
            # After calibration, ask to save settings
            save = input("Save coefficient scaling settings? (y/n): ")
            if save.lower() == 'y':
                visualizer.save_coefficient_scaling()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python script.py                  - Run the program normally")
            print("  python script.py --calibrate      - Enter calibration mode")
            print("  python script.py --help           - Show this help message")
    else:
        visualizer.run(lidar1_script, lidar2_script)
      
