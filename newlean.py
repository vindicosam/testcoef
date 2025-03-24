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
        self.lidar2_pos = (204.0, 223.5)  # Adjusted based on calibration

        # LIDAR configurations - refined based on calibration data
        self.lidar1_rotation = 103.0  # Adjusted angle
        self.lidar2_rotation = 64.0  # Adjusted angle
        self.lidar1_mirror = True
        self.lidar2_mirror = True
        self.lidar1_offset = 4.5  # Adjusted offset
        self.lidar2_offset = 0.5  # Adjusted offset

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

        # ROI Settings and Pixel-to-mm Mapping for Camera 1 (front camera)
        self.camera1_roi_top = 238  # Top of the ROI
        self.camera1_roi_bottom = 280  # Bottom of the ROI
        self.camera1_pixel_to_mm_x = (170 - (-170)) / (563 - 263)  # Calibrated conversion
        self.camera1_x_offset = -1.5  # Small offset to account for systematic error

        # ROI Settings and Pixel-to-mm Mapping for Camera 2 (left side camera)
        self.camera2_roi_top = 131  # Top of the ROI as specified
        self.camera2_roi_bottom = 190  # Bottom of the ROI as specified
        self.camera2_pixel_to_mm_y = (170 - (-170)) / (609 - 2)  # Similar conversion to camera1
        self.camera2_y_offset = -1.5  # Small offset to account for systematic error
        
        # Detection persistence for both cameras
        self.last_valid_detection1 = {"dart_mm_x": None, "dart_angle": None}
        self.last_valid_detection2 = {"dart_mm_y": None, "dart_angle": None}
        self.detection_persistence_counter1 = 0
        self.detection_persistence_counter2 = 0
        self.detection_persistence_frames = 30

        # Camera background subtractors with sensitive parameters
        self.camera1_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )
        self.camera2_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
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
        self.dartboard_image = mpimg.imread(
            "winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg"
        )

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

        # Running flag
        self.running = True

        # Calibration correction matrix based on provided screenshots
        self.calibration_points = {
            (0, 0): (-1.0, 1.5),  # Bullseye adjustment
            (-23, 167): (2.0, -4.5),  # Inner double correction
            (167, 23): (-1.5, -2.5),  # Outer double correction
            (-167, 23): (3.5, 2.5),  # Board edge correction
            (23, 167): (2.0, -3.0),  # Board edge correction
        }
        
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

        # Modified: Reduce maximum adjustment to 8mm as requested
        self.side_lean_max_adjustment = 8.0  # mm, maximum adjustment for side lean
        self.forward_lean_max_adjustment = 8.0  # mm, maximum adjustment for forward lean

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

    def start_lidar(self, script_path, queue_obj, lidar_id):
        """Start LIDAR subprocess and process data."""
        try:
            process = subprocess.Popen([script_path], stdout=subprocess.PIPE, text=True)
            print(f"LIDAR {lidar_id} started successfully.")
            data_received = False
            while self.running:
                line = process.stdout.readline()
                if line.strip():  # If line is not empty
                    if not data_received:
                        print(f"LIDAR {lidar_id} first data: {line.strip()}")
                        data_received = True
                    
                    if "a:" in line and "d:" in line:
                        try:
                            parts = line.strip().split()
                            angle = float(parts[1].replace("a:", ""))
                            distance = float(parts[2].replace("d:", ""))
                            queue_obj.put((angle, distance))
                            if not data_received:
                                print(f"LIDAR {lidar_id} processed data: angle={angle}, distance={distance}")
                        except ValueError as e:
                            print(f"LIDAR {lidar_id} value error: {e} in line: {line.strip()}")
                            continue
                        except IndexError as e:
                            print(f"LIDAR {lidar_id} index error: {e} in line: {line.strip()}")
                            continue
        except Exception as e:
            print(f"Error with LIDAR {lidar_id}: {e}")

    # IMPROVED: Enhanced dart angle calculation using the second script's approach
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
        self.camera1_stream = CameraStream(0).start()
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

            # Background subtraction and thresholding with enhanced parameters
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            fg_mask = self.camera1_bg_subtractor.apply(gray)

            # More sensitive threshold
            fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)[1]

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
                lowest_point = (-1, -1)

                for contour in contours:
                    if (
                        cv2.contourArea(contour) > 50
                    ):  # Reduced threshold to catch smaller darts
                        for point in contour:
                            x, y = point[0]
                            if tip_contour is None or y < lowest_point[1]:
                                lowest_point = (x, y)
                                tip_contour = contour

                if tip_contour is not None:
                    # IMPROVED: Use enhanced dart angle calculation
                    dart_angle, additional_info = self.calculate_dart_angle(tip_contour, fg_mask)

                    # Map pixels to mm coordinates with corrected mapping
                    tip_pixel_x = lowest_point[0]
                    dart_mm_x = (
                        180
                        - (tip_pixel_x - 126) * self.camera1_pixel_to_mm_x
                        + self.camera1_x_offset
                    )

                    # Save data
                    self.camera1_data["dart_mm_x"] = dart_mm_x
                    self.camera1_data["dart_angle"] = dart_angle

                # Update persistence
                self.last_valid_detection1 = self.camera1_data.copy()
                self.detection_persistence_counter1 = (
                    self.detection_persistence_frames
                )

            # If no dart detected but we have a valid previous detection
            elif self.detection_persistence_counter1 > 0:
                self.detection_persistence_counter1 -= 1
                if self.detection_persistence_counter1 > 0:
                    self.camera1_data = self.last_valid_detection1.copy()

        if self.camera1_stream:
            self.camera1_stream.stop()

    def camera2_detection(self):
        """Detect dart tip using the left side camera (Camera 2)."""
        self.camera2_stream = CameraStream(2).start()
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
            fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)[1]

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
                leftmost_point = (-1, -1)

                for contour in contours:
                    if cv2.contourArea(contour) > 50:
                        for point in contour:
                            x, y = point[0]
                            if tip_contour is None or x < leftmost_point[0]:
                                leftmost_point = (x, y)
                                tip_contour = contour

                if tip_contour is not None:
                    # IMPROVED: Use enhanced dart angle calculation
                    dart_angle, additional_info = self.calculate_dart_angle(tip_contour, fg_mask)

                    # Map pixels to mm coordinates
                    tip_pixel_y = leftmost_point[1]
                    # For side camera, we're detecting Y coordinate of the dart
                    dart_mm_y = (
                        180
                        - (tip_pixel_y - 126) * self.camera2_pixel_to_mm_y
                        + self.camera2_y_offset
                    )

                    # Save data
                    self.camera2_data["dart_mm_y"] = dart_mm_y
                    self.camera2_data["dart_angle"] = dart_angle

                    # Update persistence
                    self.last_valid_detection2 = self.camera2_data.copy()
                    self.detection_persistence_counter2 = (
                        self.detection_persistence_frames
                    )

            # If no dart detected but we have a valid previous detection
            elif self.detection_persistence_counter2 > 0:
                self.detection_persistence_counter2 -= 1
                if self.detection_persistence_counter2 > 0:
                    self.camera2_data = self.last_valid_detection2.copy()

        if self.camera2_stream:
            self.camera2_stream.stop()

    # IMPROVED: Enhanced side-to-side lean detection using camera data as a primary source
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

    # IMPROVED: Enhanced forward/backward lean detection using side camera data
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

    # IMPROVED: Modified projection method with camera-centric corrections
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

        Args:
            lidar_point: (x, y) position of LIDAR detection
            lidar_height: height of the LIDAR beam above board in mm
            forward_lean_angle: angle of forward/backward lean in degrees (0° = vertical)
            side_lean_angle: angle of side-to-side lean in degrees (0° = vertical)
            camera1_x: X-coordinate from front camera detection
            camera2_y: Y-coordinate from side camera detection

        Returns:
            (x, y) position of the adjusted point
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

        # Apply side-to-side lean adjustment if we have camera data
        adjusted_x = original_x
        if camera1_x is not None:
            # For side-to-side lean, we primarily use camera1 (front camera)
            # Calculate side-to-side lean adjustment
            side_lean_factor = abs(side_lean_angle) / self.MAX_SIDE_LEAN

            # Calculate X displacement (how far LIDAR point is from camera line)
            x_displacement = original_x - camera1_x

            # Apply side-to-side adjustment proportional to lean angle, with constraints
            # IMPROVED: Limited to 8mm maximum adjustment
            MAX_SIDE_ADJUSTMENT = self.side_lean_max_adjustment  # 8mm maximum
            side_adjustment = min(
                side_lean_factor * abs(x_displacement), MAX_SIDE_ADJUSTMENT
            )
            side_adjustment *= -1 if x_displacement > 0 else 1

            # Apply side-to-side adjustment
            adjusted_x = original_x + side_adjustment

        # Apply forward/backward lean adjustment
        adjusted_y = original_y
        if camera2_y is not None:
            # For forward/backward lean, we primarily use camera2 (side camera)
            # Calculate forward/backward adjustment
            forward_lean_factor = abs(forward_lean_angle) / self.MAX_FORWARD_LEAN

            # Calculate Y displacement
            y_displacement = original_y - camera2_y

            # Apply forward/backward adjustment
            # IMPROVED: Limited to 8mm maximum adjustment
            MAX_FORWARD_ADJUSTMENT = self.forward_lean_max_adjustment  # 8mm maximum
            forward_adjustment = min(
                forward_lean_factor * abs(y_displacement), MAX_FORWARD_ADJUSTMENT
            )
            forward_adjustment *= -1 if y_displacement > 0 else 1

            # Apply forward/backward adjustment
            adjusted_y = original_y + forward_adjustment

        return (adjusted_x, adjusted_y)

    def find_camera1_board_intersection(self, camera_x):
        """Calculate where the front camera epipolar line intersects the board surface."""
        if camera_x is None:
            return None

        # Simple case - the X coordinate is directly from camera, Y is 0 (board surface)
        return (camera_x, 0)

    def find_camera2_board_intersection(self, camera_y):
        """Calculate where the side camera epipolar line intersects the board surface."""
        if camera_y is None:
            return None

        # For side camera, the Y coordinate is directly from camera, X is -170 (left edge)
        return (-170, camera_y)

    def polar_to_cartesian(self, angle, distance, origin, rotation, mirror):
        """Convert polar coordinates to Cartesian."""
        if distance <= 0:
            return None, None
        angle_rad = np.radians(angle + rotation)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        if mirror:
            x = -x
        return x + origin[0], y + origin[1]

    def filter_points_by_radii(self, x, y):
        """
        Filter points based on predefined radii zones.

        Args:
            x, y: Coordinates to filter

        Returns:
            (is_valid, zone_name): Tuple with boolean indicating if point is valid and the zone name
        """
        # Calculate distance from center
        distance_from_center = np.sqrt(x**2 + y**2)

        # Check if point is within the outer board radius (including miss zone)
        if distance_from_center <= self.radii["board_edge"]:
            # Determine which zone the point is in
            zone_name = None
            for name, radius in self.radii.items():
                if distance_from_center <= radius:
                    zone_name = name
                    break

            return True, zone_name

        return False, None
    
    def apply_calibration_correction(self, x, y):
        """
        Apply calibration correction based on the calibration points map.

        Args:
            x, y: Original coordinates

        Returns:
            (corrected_x, corrected_y): Corrected coordinates
        """
        # Find the closest calibration point
        closest_point = None
        min_distance = float("inf")

        for cal_point, correction in self.calibration_points.items():
            cal_x, cal_y = cal_point
            distance = np.sqrt((x - cal_x) ** 2 + (y - cal_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_point = cal_point

        # If we found a close calibration point, apply the correction
        if closest_point is not None and min_distance < 50:  # Only apply if within 50mm
            correction_x, correction_y = self.calibration_points[closest_point]
            return x + correction_x, y + correction_y

        return x, y

    # IMPROVED: Enhanced fusion of camera and LIDAR data with camera data taking precedence
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

    def update_lean_visualization(
        self, side_lean_angle, forward_lean_angle, forward_confidence, side_confidence
    ):
        """Update the visualization of lean angles from both cameras and LIDARs."""
        # Handle None values for lean angles
        if side_lean_angle is None:
            side_lean_angle = 0.0  # Default to vertical
        if forward_lean_angle is None:
            forward_lean_angle = 0.0
        if forward_confidence is None:
            forward_confidence = 0.0
        if side_confidence is None:
            side_confidence = 0.0

        # Update text for lean angles
        self.lean_text.set_text(
            f"Side Lean: {side_lean_angle:.1f}° (conf: {side_confidence:.2f})\n"
            f"Forward Lean: {forward_lean_angle:.1f}° (conf: {forward_confidence:.2f})"
        )
        # Visualize forward/backward lean with an arrow
        if forward_confidence > 0.6 and abs(forward_lean_angle) > 5:
            # Create an arrow showing the forward lean direction
            arrow_length = 50  # Length of arrow

            # Calculate arrow endpoint for forward lean
            if forward_lean_angle > 0:
                # Leaning toward LIDAR 1 (left)
                arrow_dx = -arrow_length * np.sin(np.radians(forward_lean_angle))
                arrow_dy = arrow_length * np.cos(np.radians(forward_lean_angle))
            else:
                # Leaning toward LIDAR 2 (right)
                arrow_dx = arrow_length * np.sin(np.radians(-forward_lean_angle))
                arrow_dy = arrow_length * np.cos(np.radians(-forward_lean_angle))

            # Add or update forward lean arrow annotation
            if (
                hasattr(self, "forward_lean_arrow")
                and self.forward_lean_arrow is not None
            ):
                self.forward_lean_arrow.remove()
            self.forward_lean_arrow = self.ax.arrow(
                0,
                0,
                arrow_dx,
                arrow_dy,
                width=3,
                head_width=10,
                head_length=10,
                fc="purple",
                ec="purple",
                alpha=0.7,
            )

            # Add text label near arrow
            if (
                hasattr(self, "forward_arrow_text")
                and self.forward_arrow_text is not None
            ):
                self.forward_arrow_text.remove()
            self.forward_arrow_text = self.ax.text(
                arrow_dx / 2,
                arrow_dy / 2,
                f"F: {abs(forward_lean_angle):.1f}°",
                color="purple",
                fontsize=9,
                ha="center",
                va="center",
            )
        else:
            # Remove arrow if lean not detected with confidence
            if (
                hasattr(self, "forward_lean_arrow")
                and self.forward_lean_arrow is not None
            ):
                self.forward_lean_arrow.remove()
                self.forward_lean_arrow = None
            if (
                hasattr(self, "forward_arrow_text")
                and self.forward_arrow_text is not None
            ):
                self.forward_arrow_text.remove()
                self.forward_arrow_text = None

        # Visualize side-to-side lean with an arrow
        if side_confidence > 0.6 and abs(side_lean_angle) > 5:
            # Create an arrow showing the side lean direction
            arrow_length = 50  # Length of arrow

            # Calculate arrow endpoint for side lean
            if side_lean_angle > 0:
                # Leaning toward right
                arrow_dx = arrow_length * np.sin(np.radians(side_lean_angle))
                arrow_dy = 0
            else:
                # Leaning toward left
                arrow_dx = arrow_length * np.sin(np.radians(side_lean_angle))
                arrow_dy = 0

            # Add or update side lean arrow annotation
            if hasattr(self, "side_lean_arrow") and self.side_lean_arrow is not None:
                self.side_lean_arrow.remove()
            self.side_lean_arrow = self.ax.arrow(
                0,
                0,
                arrow_dx,
                arrow_dy,
                width=3,
                head_width=10,
                head_length=10,
                fc="cyan",
                ec="cyan",
                alpha=0.7,
            )

            # Add text label near arrow
            if hasattr(self, "side_arrow_text") and self.side_arrow_text is not None:
                self.side_arrow_text.remove()
            self.side_arrow_text = self.ax.text(
                arrow_dx / 2,
                arrow_dy + 15,
                f"S: {abs(side_lean_angle):.1f}°",
                color="cyan",
                fontsize=9,
                ha="center",
                va="center",
            )
        else:
            # Remove arrow if lean not detected with confidence
            if hasattr(self, "side_lean_arrow") and self.side_lean_arrow is not None:
                self.side_lean_arrow.remove()
                self.side_lean_arrow = None
            if hasattr(self, "side_arrow_text") and self.side_arrow_text is not None:
                self.side_arrow_text.remove()
                self.side_arrow_text = None

    def update_plot(self, frame):
        """Update plot data with enhanced 3D lean correction using dual cameras."""
        x1, y1 = [], []
        x2, y2 = [], []

        # Track the most significant LIDAR points for vector visualization
        lidar1_most_significant = None
        lidar2_most_significant = None

        # Process LIDAR 1 data
        while not self.lidar1_queue.empty():
            angle, distance = self.lidar1_queue.get()
            x, y = self.polar_to_cartesian(
                angle,
                distance - self.lidar1_offset,
                self.lidar1_pos,
                self.lidar1_rotation,
                self.lidar1_mirror,
            )

            if x is not None:
                is_valid, zone = self.filter_points_by_radii(x, y)
                if is_valid:
                    # Store point for recent history
                    self.lidar1_recent_points.append((x, y))
                    if len(self.lidar1_recent_points) > self.max_recent_points:
                        self.lidar1_recent_points.pop(0)

                    x1.append(x)
                    y1.append(y)

                    # Update most significant point (closest to center)
                    dist_from_center = np.sqrt(x**2 + y**2)
                    if (
                        lidar1_most_significant is None
                        or dist_from_center < lidar1_most_significant[2]
                    ):
                        lidar1_most_significant = (x, y, dist_from_center)

        # Process LIDAR 2 data
        while not self.lidar2_queue.empty():
            angle, distance = self.lidar2_queue.get()
            x, y = self.polar_to_cartesian(
                angle,
                distance - self.lidar2_offset,
                self.lidar2_pos,
                self.lidar2_rotation,
                self.lidar2_mirror,
            )

            if x is not None:
                is_valid, zone = self.filter_points_by_radii(x, y)
                if is_valid:
                    # Store point for recent history
                    self.lidar2_recent_points.append((x, y))
                    if len(self.lidar2_recent_points) > self.max_recent_points:
                        self.lidar2_recent_points.pop(0)

                    x2.append(x)
                    y2.append(y)

                    # Update most significant point (closest to center)
                    dist_from_center = np.sqrt(x**2 + y**2)
                    if (
                        lidar2_most_significant is None
                        or dist_from_center < lidar2_most_significant[2]
                    ):
                        lidar2_most_significant = (x, y, dist_from_center)

        # Calculate LIDAR average positions if enough data
        lidar1_avg = None
        lidar2_avg = None

        if len(self.lidar1_recent_points) > 0:
            avg_x = sum(p[0] for p in self.lidar1_recent_points) / len(
                self.lidar1_recent_points
            )
            avg_y = sum(p[1] for p in self.lidar1_recent_points) / len(
                self.lidar1_recent_points
            )
            lidar1_avg = (avg_x, avg_y)

        if len(self.lidar2_recent_points) > 0:
            avg_x = sum(p[0] for p in self.lidar2_recent_points) / len(
                self.lidar2_recent_points
            )
            avg_y = sum(p[1] for p in self.lidar2_recent_points) / len(
                self.lidar2_recent_points
            )
            lidar2_avg = (avg_x, avg_y)
            
        # Get camera data from both cameras
        camera1_x = self.camera1_data.get("dart_mm_x")
        camera1_lean_angle = self.camera1_data.get("dart_angle", 90)  # Default to vertical if unknown

        camera2_y = self.camera2_data.get("dart_mm_y")
        camera2_lean_angle = self.camera2_data.get("dart_angle", 90)  # Default to vertical if unknown

        # IMPROVED: Use enhanced lean detection using camera data as primary source
        forward_lean_angle = 0
        forward_confidence = 0
        side_lean_angle = 0
        side_confidence = 0

        # Detect side-to-side lean (prioritize camera1 data)
        if camera1_lean_angle is not None or (lidar1_avg is not None and lidar2_avg is not None):
            side_lean_angle, side_confidence = self.detect_side_to_side_lean(
                lidar1_avg, lidar2_avg
            )

        # Detect forward/backward lean (prioritize camera2 data)
        if camera2_lean_angle is not None or (lidar1_avg is not None and lidar2_avg is not None):
            forward_lean_angle, forward_confidence = self.detect_forward_backward_lean(
                lidar1_avg, lidar2_avg
            )

        # Add to lean history for smoothing
        self.lean_history_forward.append((forward_lean_angle, forward_confidence))
        if len(self.lean_history_forward) > self.max_lean_history:
            self.lean_history_forward.pop(0)

        self.lean_history_side.append((side_lean_angle, side_confidence))
        if len(self.lean_history_side) > self.max_lean_history:
            self.lean_history_side.pop(0)

        # Calculate weighted average of forward lean angles
        if self.lean_history_forward:
            total_weight = sum(conf for _, conf in self.lean_history_forward)
            if total_weight > 0:
                smoothed_lean = (
                    sum(angle * conf for angle, conf in self.lean_history_forward)
                    / total_weight
                )
                forward_lean_angle = smoothed_lean

        # Calculate weighted average of side lean angles
        if self.lean_history_side:
            total_weight = sum(conf for _, conf in self.lean_history_side)
            if total_weight > 0:
                smoothed_lean = (
                    sum(angle * conf for angle, conf in self.lean_history_side)
                    / total_weight
                )
                side_lean_angle = smoothed_lean

        # Update current values for visualization
        self.current_forward_lean_angle = forward_lean_angle
        self.current_side_lean_angle = side_lean_angle
        self.forward_lean_confidence = forward_confidence
        self.side_lean_confidence = side_confidence
        
        # Calculate intersection of camera vectors with board surface
        self.camera1_board_intersection = self.find_camera1_board_intersection(
            camera1_x
        )
        self.camera2_board_intersection = self.find_camera2_board_intersection(
            camera2_y
        )

        # IMPROVED: Project LIDAR points with enhanced lean correction
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None

        if lidar1_avg is not None:
            self.lidar1_projected_point = self.project_lidar_point_with_3d_lean(
                lidar1_avg,
                self.lidar1_height,
                forward_lean_angle,
                side_lean_angle,
                camera1_x,
                camera2_y,
            )

        if lidar2_avg is not None:
            self.lidar2_projected_point = self.project_lidar_point_with_3d_lean(
                lidar2_avg,
                self.lidar2_height,
                forward_lean_angle,
                side_lean_angle,
                camera1_x,
                camera2_y,
            )

        # Calculate final tip position using all available data with enhanced 3D lean correction
        final_tip_position = self.calculate_final_tip_position(
            self.camera1_board_intersection,
            self.camera2_board_intersection,
            self.lidar1_projected_point,
            self.lidar2_projected_point,
        )

        # Apply calibration correction to final tip position
        if final_tip_position is not None:
            final_tip_position = self.apply_calibration_correction(
                final_tip_position[0], final_tip_position[1]
            )

            # Update detected dart position
            self.detected_dart.set_data(
                [final_tip_position[0]], [final_tip_position[1]]
            )

            # Check which zone the dart is in
            distance_from_center = np.sqrt(
                final_tip_position[0] ** 2 + final_tip_position[1] ** 2
            )
            detected_zone = None
            for name, radius in self.radii.items():
                if distance_from_center <= radius:
                    detected_zone = name
                    break

            # Print detailed dart information with lean angles
            print(
                f"Dart detected - X: {final_tip_position[0]:.1f}, Y: {final_tip_position[1]:.1f}, "
                f"Zone: {detected_zone if detected_zone else 'Outside'}"
            )
            print(
                f"Side lean: {side_lean_angle:.1f}° (conf: {side_confidence:.2f}), "
                f"Forward lean: {forward_lean_angle:.1f}° (conf: {forward_confidence:.2f})"
            )
        else:
            self.detected_dart.set_data([], [])

        # Update the lean visualization
        self.update_lean_visualization(
            side_lean_angle, forward_lean_angle, forward_confidence, side_confidence
        )

        # Update camera 1 (front) visualization
        if self.camera1_board_intersection is not None:
            # Calculate direction vector from camera to intersection
            camera1_x = self.camera1_board_intersection[0]
            camera1_y = self.camera1_board_intersection[1]

            # Calculate unit vector from camera to intersection
            dx = camera1_x - self.camera1_position[0]
            dy = camera1_y - self.camera1_position[1]
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Normalize and scale to vector length
                unit_x = dx / length
                unit_y = dy / length
                vector_end_x = (
                    self.camera1_position[0] + self.camera1_vector_length * unit_x
                )
                vector_end_y = (
                    self.camera1_position[1] + self.camera1_vector_length * unit_y
                )

                # Draw camera vector and intersection point
                self.camera1_vector.set_data(
                    [self.camera1_position[0], vector_end_x],
                    [self.camera1_position[1], vector_end_y],
                )
                self.camera1_dart.set_data([camera1_x], [camera1_y])
            else:
                self.camera1_vector.set_data([], [])
                self.camera1_dart.set_data([], [])
        else:
            self.camera1_vector.set_data([], [])
            self.camera1_dart.set_data([], [])
            
        # Update camera 2 (left side) visualization
        if self.camera2_board_intersection is not None:
            # Calculate direction vector from camera to intersection
            camera2_x = self.camera2_board_intersection[0]
            camera2_y = self.camera2_board_intersection[1]

            # Calculate unit vector from camera to intersection
            dx = camera2_x - self.camera2_position[0]
            dy = camera2_y - self.camera2_position[1]
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Normalize and scale to vector length
                unit_x = dx / length
                unit_y = dy / length
                vector_end_x = (
                    self.camera2_position[0] + self.camera2_vector_length * unit_x
                )
                vector_end_y = (
                    self.camera2_position[1] + self.camera2_vector_length * unit_y
                )

                # Draw camera vector and intersection point
                self.camera2_vector.set_data(
                    [self.camera2_position[0], vector_end_x],
                    [self.camera2_position[1], vector_end_y],
                )
                self.camera2_dart.set_data([camera2_x], [camera2_y])
            else:
                self.camera2_vector.set_data([], [])
                self.camera2_dart.set_data([], [])
        else:
            self.camera2_vector.set_data([], [])
            self.camera2_dart.set_data([], [])

        # Update LIDAR1 visualization
        if lidar1_most_significant is not None:
            lidar1_x, lidar1_y = lidar1_most_significant[0], lidar1_most_significant[1]

            # Draw vector from LIDAR1 position to detected point
            dx = lidar1_x - self.lidar1_pos[0]
            dy = lidar1_y - self.lidar1_pos[1]
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Create 600mm vector
                unit_x = dx / length
                unit_y = dy / length
                vector_end_x = self.lidar1_pos[0] + 600 * unit_x
                vector_end_y = self.lidar1_pos[1] + 600 * unit_y

                # Draw LIDAR1 vector
                self.lidar1_vector.set_data(
                    [self.lidar1_pos[0], vector_end_x],
                    [self.lidar1_pos[1], vector_end_y],
                )

                # Draw projected point if available
                if self.lidar1_projected_point is not None:
                    self.lidar1_dart.set_data(
                        [self.lidar1_projected_point[0]],
                        [self.lidar1_projected_point[1]],
                    )
                else:
                    self.lidar1_dart.set_data([], [])
            else:
                self.lidar1_vector.set_data([], [])
                self.lidar1_dart.set_data([], [])
        else:
            self.lidar1_vector.set_data([], [])
            self.lidar1_dart.set_data([], [])
            
        # Update LIDAR2 visualization
        if lidar2_most_significant is not None:
            lidar2_x, lidar2_y = lidar2_most_significant[0], lidar2_most_significant[1]

            # Draw vector from LIDAR2 position to detected point
            dx = lidar2_x - self.lidar2_pos[0]
            dy = lidar2_y - self.lidar2_pos[1]
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Create 600mm vector
                unit_x = dx / length
                unit_y = dy / length
                vector_end_x = self.lidar2_pos[0] + 600 * unit_x
                vector_end_y = self.lidar2_pos[1] + 600 * unit_y

                # Draw LIDAR2 vector
                self.lidar2_vector.set_data(
                    [self.lidar2_pos[0], vector_end_x],
                    [self.lidar2_pos[1], vector_end_y],
                )

                # Draw projected point if available
                if self.lidar2_projected_point is not None:
                    self.lidar2_dart.set_data(
                        [self.lidar2_projected_point[0]],
                        [self.lidar2_projected_point[1]],
                    )
                else:
                    self.lidar2_dart.set_data([], [])
            else:
                self.lidar2_vector.set_data([], [])
                self.lidar2_dart.set_data([], [])
        else:
            self.lidar2_vector.set_data([], [])
            self.lidar2_dart.set_data([], [])

        # Update LIDAR scatter plots
        self.scatter1.set_data(x1, y1)
        self.scatter2.set_data(x2, y2)

        return (
            self.scatter1,
            self.scatter2,
            self.camera1_vector,
            self.camera2_vector,
            self.detected_dart,
            self.lidar1_vector,
            self.lidar2_vector,
            self.camera1_dart,
            self.camera2_dart,
            self.lidar1_dart,
            self.lidar2_dart,
        )

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
    visualizer.run(lidar1_script, lidar2_script)
