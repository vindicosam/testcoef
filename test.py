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
import csv
from datetime import datetime

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

        # Camera configuration - MODIFIED FOR LEFT POSITION
        self.camera_position = (-350, 0)     # Camera is to the left of the board
        self.camera_vector_length = 1600     # Vector length in mm
        self.camera_data = {"dart_mm_y": None, "dart_angle": None}  # Now tracking Y position

        # ROI Settings for side camera (similar to camera2 in the dual setup)
        self.camera_board_plane_y = 250  # The y-coordinate where the board surface is
        self.camera_roi_range = 30       # How much above and below to include
        self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
        self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
        self.camera_roi_left = 119   # Example left boundary
        self.camera_roi_right = 604  # Example right boundary
        
        # Add calibration point system
        self.camera_calibration_points = []  # List of (pixel_x, mm_y) tuples for calibration
        # Default linear calibration as fallback
        self.pixel_to_mm_factor = -0.628  # Slope in mm/pixel 
        self.pixel_offset = 192.8        # Board y when pixel_x = 0

        # Detection persistence to maintain visibility
        self.last_valid_detection = {"dart_mm_y": None, "dart_angle": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30

        # Camera background subtractor with improved parameters from second script
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=162,  # Reduced to adapt faster
            varThreshold=67,  # More sensitive
            detectShadows=False
        )
        
        # Frame differencing threshold from second script
        self.diff_threshold = 25
        self.min_contour_area = 30
        
        # Previous frame storage for frame differencing
        self.prev_gray = None

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
        
        # Intersection point of camera vector with board plane
        self.camera_board_intersection = None

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

        # --- Coefficient dictionaries remain the same ---
        # Coefficients for the outer single area (large segments)
        self.large_segment_coeff = {
            "14_5": {"x_correction": -1.888, "y_correction": 12.790},
            "11_4": {"x_correction": -6.709, "y_correction": 14.045},
            # ... [rest of the coefficients remain unchanged]
        }
        
        # [Other coefficient dictionaries remain unchanged]

        # Calibration factors for lean correction
        self.side_lean_max_adjustment = 6.0  # mm, maximum adjustment for side lean
        self.up_down_lean_max_adjustment = 4.0  # mm, renamed from forward_lean_max_adjustment
        
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
        self.current_up_down_lean_angle = 0.0  # Renamed from current_forward_lean_angle
        self.up_down_lean_confidence = 0.0  # Renamed from forward_lean_confidence
        self.lean_history = []  # Store recent lean readings for smoothing
        self.max_lean_history = 60  # Keep track of last 60 lean readings
        self.lean_arrow = None  # For visualization
        self.arrow_text = None  # For visualization text
        
        # Maximum expected lean angles
        self.MAX_SIDE_LEAN = 35.0  # Maximum expected side-to-side lean in degrees
        self.MAX_UP_DOWN_LEAN = 30.0  # Renamed from MAX_FORWARD_LEAN
        
        # Maximum expected X-difference for maximum lean (calibration parameter)
        self.MAX_X_DIFF_FOR_MAX_LEAN = 4.0  # mm
        
        # CSV logging
        self.initialize_csv_logging()
        
        # Setup visualization
        self.setup_plot()

        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)

    def initialize_csv_logging(self):
        """Initialize CSV file for logging dart data."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"dart_data_{timestamp}.csv"
        
        # Create CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Dart_X_mm', 'Dart_Y_mm', 
                                'Tip_Pixel_X', 'Tip_Pixel_Y', 
                                'Side_Lean_Angle', 'Up_Down_Lean_Angle'])
        
        print(f"CSV logging initialized: {self.csv_filename}")

    def log_dart_data(self, final_tip_position, tip_pixel, side_lean_angle, up_down_lean_angle):
        """Log dart data to CSV file."""
        if final_tip_position is None:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                timestamp,
                f"{final_tip_position[0]:.2f}" if final_tip_position else "None",
                f"{final_tip_position[1]:.2f}" if final_tip_position else "None",
                f"{tip_pixel[0]}" if tip_pixel else "None",
                f"{tip_pixel[1]}" if tip_pixel else "None",
                f"{side_lean_angle:.2f}" if side_lean_angle is not None else "None",
                f"{up_down_lean_angle:.2f}" if up_down_lean_angle is not None else "None"
            ])

    def add_calibration_point(self, pixel_x, mm_y):
        """Add a calibration point mapping pixel x-coordinate to mm y-coordinate."""
        self.camera_calibration_points.append((pixel_x, mm_y))
        # Sort by pixel_x for proper interpolation
        self.camera_calibration_points.sort(key=lambda p: p[0])
        print(f"Added calibration point: pixel_x={pixel_x}, mm_y={mm_y}")
        print(f"Current calibration points: {self.camera_calibration_points}")

    def pixel_to_mm(self, pixel_x):
        """
        Convert pixel x-coordinate to mm y-coordinate using calibration points.
        Uses linear interpolation between known points or linear equation as fallback.
        """
        # If we have at least 2 calibration points, use interpolation
        if len(self.camera_calibration_points) >= 2:
            # Find the two nearest calibration points for interpolation
            for i in range(len(self.camera_calibration_points) - 1):
                p1_pixel, p1_mm = self.camera_calibration_points[i]
                p2_pixel, p2_mm = self.camera_calibration_points[i + 1]
                
                # If pixel_x is between these two points, interpolate
                if p1_pixel <= pixel_x <= p2_pixel:
                    # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                    return p1_mm + (pixel_x - p1_pixel) * (p2_mm - p1_mm) / (p2_pixel - p1_pixel)
                    
            # If outside the calibration range, use the closest calibration point
            if pixel_x < self.camera_calibration_points[0][0]:
                return self.camera_calibration_points[0][1]
            else:
                return self.camera_calibration_points[-1][1]
        else:
           # Fallback to linear equation if not enough calibration points
            return self.pixel_to_mm_factor * pixel_x + self.pixel_offset

    def clear_calibration_points(self):
        """Clear all calibration points."""
        self.camera_calibration_points = []
        print("Calibration points cleared.")

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

    def setup_plot(self):
        """Initialize the plot with enhanced visualization for 3D lean."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_title("LIDAR and Camera Vector Visualization")
        self.ax.grid(True)

        # Add dartboard image
        self.update_dartboard_image()

        # Plot LIDAR positions
        self.ax.plot(*self.lidar1_pos, "bo", label="LIDAR 1")
        self.ax.plot(*self.lidar2_pos, "go", label="LIDAR 2")
        self.ax.plot(*self.camera_position, "ro", label="Camera")
        
        # Draw all radii circles
        for name, radius in self.radii.items():
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray', alpha=0.4)
            self.ax.add_patch(circle)
            self.ax.text(0, radius, name, color='gray', fontsize=8, ha='center', va='bottom')

        # Vectors and detected dart position
        self.scatter1, = self.ax.plot([], [], "b.", label="LIDAR 1 Data", zorder=3)
        self.scatter2, = self.ax.plot([], [], "g.", label="LIDAR 2 Data", zorder=3)
        self.camera_vector, = self.ax.plot([], [], "r--", label="Camera Vector")
        self.lidar1_vector, = self.ax.plot([], [], "b--", label="LIDAR 1 Vector")
        self.lidar2_vector, = self.ax.plot([], [], "g--", label="LIDAR 2 Vector")
        self.camera_dart, = self.ax.plot([], [], "rx", markersize=8, label="Camera Intersection")
        self.lidar1_dart, = self.ax.plot([], [], "bx", markersize=8, label="LIDAR 1 Projected", zorder=3)
        self.lidar2_dart, = self.ax.plot([], [], "gx", markersize=8, label="LIDAR 2 Projected", zorder=3)
        self.detected_dart, = self.ax.plot([], [], "ro", markersize=4, label="Final Tip Position", zorder=10)
        
        # Add text annotation for lean angles
        self.lean_text = self.ax.text(-380, 380, "", fontsize=9)
        
        self.ax.legend(loc="upper right", fontsize=8)

    def update_dartboard_image(self):
        """Update the dartboard image extent."""
        scaled_extent = [-170 * self.board_scale_factor, 170 * self.board_scale_factor,
                         -170 * self.board_scale_factor, 170 * self.board_scale_factor]
        self.ax.imshow(self.dartboard_image, extent=scaled_extent, zorder=0)

    def start_lidar(self, script_path, queue_obj, lidar_id):
        """Start LIDAR subprocess and process data."""
        try:
            process = subprocess.Popen([script_path], stdout=subprocess.PIPE, text=True)
            print(f"LIDAR {lidar_id} started successfully.")
            while self.running:
                line = process.stdout.readline()
                if "a:" in line and "d:" in line:
                    try:
                        parts = line.strip().split()
                        angle = float(parts[1].replace("a:", ""))
                        distance = float(parts[2].replace("d:", ""))
                        queue_obj.put((angle, distance))
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Error with LIDAR {lidar_id}: {e}")

    def measure_tip_angle(self, mask, tip_point):
        """
        Measure the angle of the dart tip using the approach from the second script.
        Enhanced accuracy with RANSAC fitting.
        
        Args:
            mask: Binary mask containing dart
            tip_point: Detected tip coordinates (x,y)
            
        Returns:
            angle: Angle in degrees (90Â° = vertical) or None if angle couldn't be calculated
        """
        if tip_point is None:
            return None
            
        tip_x, tip_y = tip_point
        
        # Define search parameters
        search_depth = 25  # How far to search from the tip
        search_width = 40  # Width of search area
        min_points = 6     # Reduced from 8 to be more tolerant with fast movement
        
        # Define region to search for the dart shaft
        # For a left camera, search to the right of the tip point
        min_x = max(0, tip_x)
        max_x = min(mask.shape[1] - 1, tip_x + search_depth)
        min_y = max(0, tip_y - search_width)
        max_y = min(mask.shape[0] - 1, tip_y + search_width)
        
        # Find all white pixels in the search area
        points_right = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if mask[y, x] > 0:  # White pixel
                    points_right.append((x, y))
        
        if len(points_right) < min_points:  # Need enough points for a good fit
            return None
            
        # Use RANSAC for more robust angle estimation
        # This helps ignore outlier points that may come from noise
        best_angle = None
        best_inliers = 0
        
        for _ in range(10):  # Try several random samples
            if len(points_right) < 2:
                continue
                
            # Randomly select two points
            indices = np.random.choice(len(points_right), 2, replace=False)
            p1 = points_right[indices[0]]
            p2 = points_right[indices[1]]
            
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
            for point in points_right:
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
        
        if best_angle is None:
            # Fall back to simple linear regression if RANSAC fails
            points = np.array(points_right)
            if len(points) < 2:
                return None
                
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
        
        return best_angle

    def camera_detection(self):
        """
        Detect dart tip using the camera (left position) with improved settings from the second script.
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Wait for camera initialization
        time.sleep(1)
        
        # Initialize previous frame for frame differencing
        self.prev_gray = None

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Rotate frame 180 degrees since camera is now on the left
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Extract ROI as horizontal strip at board surface level
            roi = frame[self.camera_roi_top:self.camera_roi_bottom, 
                         self.camera_roi_left:self.camera_roi_right]

            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Frame differencing (from second script)
            if self.prev_gray is None or self.prev_gray.shape != gray.shape:
                # Initialize or reset previous frame if shapes don't match
                self.prev_gray = gray.copy()
            
            # Calculate absolute difference between current and previous frame
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            _, diff_thresh = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            
            # Update previous frame for next iteration
            self.prev_gray = gray.copy()
            
            # Background subtraction
            fg_mask = self.camera_bg_subtractor.apply(gray)
            
            # Lower threshold for foreground detection
            fg_mask = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY)[1]  # Reduced from 180
            
            # Combine background subtraction with frame differencing to catch fast movement
            combined_mask = cv2.bitwise_or(fg_mask, diff_thresh)
            
            # Morphological operations to enhance the dart
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)  # Increased from 1

            # Reset current detection
            self.camera_data["dart_mm_y"] = None
            self.camera_data["dart_angle"] = None

            # Detect contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the rightmost point since camera is now on the left
                tip_contour = None
                tip_point = None
                
                for contour in contours:
                    if cv2.contourArea(contour) > self.min_contour_area:  # Reduced threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        dart_pixel_x = x + w // 2  # Center x of contour
                        
                        # Use the board plane as the y-position
                        roi_center_y = self.camera_board_plane_y - self.camera_roi_top
                        
                        if tip_contour is None:
                            tip_contour = contour
                            tip_point = (dart_pixel_x, roi_center_y)
                
                if tip_contour is not None and tip_point is not None:
                    # Calculate dart angle
                    dart_angle = self.measure_tip_angle(combined_mask, tip_point)
                    
                    # Map pixels to mm coordinates using new conversion
                    # Convert to global pixel coordinates
                    global_pixel_x = tip_point[0] + self.camera_roi_left
                    global_pixel_y = tip_point[1] + self.camera_roi_top
                    
                    dart_mm_y = self.pixel_to_mm(global_pixel_x)
                    
                    # Save data
                    self.camera_data["dart_mm_y"] = dart_mm_y
                    self.camera_data["dart_angle"] = dart_angle
                    self.camera_data["tip_pixel"] = (global_pixel_x, global_pixel_y)  # Store for CSV logging
                    
                    # Update persistence
                    self.last_valid_detection = self.camera_data.copy()
                    self.detection_persistence_counter = self.detection_persistence_frames
            
            # If no dart detected but we have a valid previous detection
            elif self.detection_persistence_counter > 0:
                self.detection_persistence_counter -= 1
                if self.detection_persistence_counter > 0:
                    self.camera_data = self.last_valid_detection.copy()
        
        cap.release()

    def polar_to_cartesian(self, angle, distance, lidar_pos, rotation, mirror):
        """Convert polar coordinates to Cartesian."""
        if distance <= 0:
            return None, None
        angle_rad = np.radians(angle + rotation)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        if mirror:
            x = -x
        return x + lidar_pos[0], y + lidar_pos[1]

    def filter_points_by_radii(self, x, y):
        """Check if a point falls within any radii."""
        distance = np.sqrt(x**2 + y**2)
        for name, radius in self.radii.items():
            if distance <= radius:
                return True, name
        return False, None

    def apply_calibration_correction(self, x, y):
        """Apply improved calibration correction using weighted interpolation."""
        if not self.calibration_points:
            return x, y
            
        # Calculate inverse distance weighted average of corrections
        total_weight = 0
        correction_x_weighted = 0
        correction_y_weighted = 0
        
        for ref_point, correction in self.calibration_points.items():
            ref_x, ref_y = ref_point
            dist = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
            
            # Avoid division by zero and use inverse square for more local influence
            if dist < 0.1:
                # Very close to a reference point, use its correction directly
                return x + correction[0], y + correction[1]
            
            # Use inverse square weighting with a maximum influence range
            weight = 1 / (dist * dist) if dist < 100 else 0
            
            correction_x_weighted += correction[0] * weight
            correction_y_weighted += correction[1] * weight
            total_weight += weight
        
        # Apply weighted corrections if we have valid weights
        if total_weight > 0:
            return x + (correction_x_weighted / total_weight), y + (correction_y_weighted / total_weight)
        
        return x, y

    def get_wire_proximity_factor(self, x, y):
        """
        Calculate a proximity factor (0.0 to 1.0) indicating how close a point is to a wire.
        1.0 means directly on a wire, 0.0 means far from any wire.
        
        Args:
            x: x-coordinate
            y: y-coordinate
            
        Returns:
            float: Proximity factor from 0.0 to 1.0
        """
        # Calculate distance from center
        dist = math.sqrt(x*x + y*y)
        
        # Check proximity to circular wires (ring boundaries)
        ring_wire_threshold = 5.0  # mm
        min_ring_distance = float('inf')
        for radius in [self.radii["bullseye"], self.radii["outer_bull"], 
                      self.radii["inner_treble"], self.radii["outer_treble"],
                      self.radii["inner_double"], self.radii["outer_double"]]:
            ring_distance = abs(dist - radius)
            min_ring_distance = min(min_ring_distance, ring_distance)
        
        # Check proximity to radial wires (segment boundaries)
        segment_wire_threshold = 5.0  # mm
        
        # Calculate angle in degrees
        angle_deg = math.degrees(math.atan2(y, x))
        if angle_deg < 0:
            angle_deg += 360
        
        # Check proximity to segment boundaries (every 18 degrees)
        min_segment_distance = float('inf')
        segment_boundary_degree = 9  # The first boundary is at 9 degrees, then every 18 degrees
        for i in range(20):  # 20 segments
            boundary_angle = (segment_boundary_degree + i * 18) % 360
            angle_diff = min(abs(angle_deg - boundary_angle), 360 - abs(angle_deg - boundary_angle))
            
            # Convert angular difference to mm at this radius
            linear_diff = angle_diff * math.pi / 180 * dist
            min_segment_distance = min(min_segment_distance, linear_diff)
        
        # Get the minimum distance to any wire
        min_wire_distance = min(min_ring_distance, min_segment_distance)
        
        # Convert to a factor from 0.0 to 1.0
        # 1.0 means on the wire, 0.0 means at or beyond the threshold distance
        if min_wire_distance >= ring_wire_threshold:
            return 0.0
        else:
            return 1.0 - (min_wire_distance / ring_wire_threshold)

    def apply_segment_coefficients(self, x, y):
        """
        Apply segment-specific coefficients based on the ring (area) of the dart,
        weighted by proximity to wires and segment-specific scaling factors.
        Corrections are only applied to darts near wires.
        
        Args:
            x: x-coordinate of dart position
            y: y-coordinate of dart position
            
        Returns:
            Tuple of (corrected_x, corrected_y)
        """
        # Get wire proximity factor (0.0 to 1.0)
        wire_factor = self.get_wire_proximity_factor(x, y)
        
        # If far from any wire, return original position
        if wire_factor <= 0.0:
            return x, y
        
        dist = math.sqrt(x*x + y*y)
        
        # No correction for bullseye and outer bull
        if dist <= self.radii["outer_bull"]:
            return x, y

        # Determine segment number from angle (using offset of 9° as before)
        angle_deg = math.degrees(math.atan2(y, x))
        if angle_deg < 0:
            angle_deg += 360
        segment = int(((angle_deg + 9) % 360) / 18) + 1
        seg_str = str(segment)

        # Get the scaling factor specific to this segment and ring type
        ring_type = ""
        
        # Choose coefficient dictionary based on ring and set ring type for scaling
        if dist < self.radii["inner_treble"]:
            # Inner single area (small segment)
            coeff_dict = self.small_segment_coeff
            ring_type = "small"
            # Try several candidate keys
            possible_keys = [f"{seg_str}_1", f"{seg_str}_0", f"{seg_str}_5", f"{seg_str}_4"]
        elif dist < self.radii["outer_treble"]:
            # Triple ring area
            coeff_dict = self.trebles_coeff
            ring_type = "trebles"
            possible_keys = [f"{seg_str}_1", f"{seg_str}_5", f"{seg_str}_0", f"{seg_str}_3", f"{seg_str}_2"]
        elif dist < self.radii["inner_double"]:
            # Outer single area (large segment)
            coeff_dict = self.large_segment_coeff
            ring_type = "large"
            possible_keys = [f"{seg_str}_5", f"{seg_str}_4", f"{seg_str}_0", f"{seg_str}_1"]
        elif dist < self.radii["outer_double"]:
            # Double ring area
            coeff_dict = self.doubles_coeff
            ring_type = "doubles"
            possible_keys = [f"{seg_str}_1", f"{seg_str}_5", f"{seg_str}_0", f"{seg_str}_3"]
        else:
            # Outside board edge or miss, no correction
            return x, y
            
        # Get the segment-specific scaling factor for this ring type
        scaling_factor = 1.0
        if segment in self.coefficient_scaling and ring_type in self.coefficient_scaling[segment]:
            scaling_factor = self.coefficient_scaling[segment][ring_type]

        # Look up the first matching key in the selected dictionary
        for key in possible_keys:
            if key in coeff_dict:
                coeff = coeff_dict[key]
                # Apply correction weighted by wire proximity factor AND segment-specific scaling
                correction_factor = wire_factor * scaling_factor
                return (x + (coeff["x_correction"] * correction_factor), 
                        y + (coeff["y_correction"] * correction_factor))
        return x, y

    def detect_up_down_lean(self, lidar1_point, lidar2_point):
        """
        Enhanced up/down lean detection that works with one or two LIDARs.
        
        Args:
            lidar1_point: (x, y) position from LIDAR 1
            lidar2_point: (x, y) position from LIDAR 2
            
        Returns:
            lean_angle: Estimated lean angle in degrees (0° = vertical, positive = up, negative = down)
            confidence: Confidence level in the lean detection (0-1)
        """
        # Case 1: Both LIDARs active - use standard X-difference calculation
        if lidar1_point is not None and lidar2_point is not None:
            # Extract x-coordinates from both LIDARs
            x1 = lidar1_point[0]
            x2 = lidar2_point[0]
            
            # Calculate x-difference as indicator of up/down lean
            x_diff = x1 - x2
            
            # Convert x-difference to an angle
            lean_angle = (x_diff / self.MAX_X_DIFF_FOR_MAX_LEAN) * self.MAX_UP_DOWN_LEAN
            
            # Calculate confidence based on how many points we have
            confidence = min(1.0, (len(self.lidar1_recent_points) + len(self.lidar2_recent_points)) / 
                             (2 * self.max_recent_points))
            
        # Case 2: Single LIDAR with significant side lean - infer up/down lean
        elif (lidar1_point is not None or lidar2_point is not None) and self.camera_data.get("dart_angle") is not None:
            side_lean_angle = self.camera_data.get("dart_angle")
            
            # Calculate how far from vertical the side lean is
            side_lean_deviation = abs(90 - side_lean_angle)
            
            # Only LIDAR 1 (upper side) active
            if lidar1_point is not None and lidar2_point is None:
                # Get quadrant information
                y_pos = lidar1_point[1]
                
                # If leaning up and on upper part: likely toward LIDAR 1
                if side_lean_angle < 80 and y_pos > 0:
                    # More side lean = more up lean
                    lean_angle = side_lean_deviation * 0.5
                # If leaning up and on lower part: likely away from LIDAR 2
                elif side_lean_angle < 80 and y_pos <= 0:
                    lean_angle = side_lean_deviation * 0.3
                # Near vertical or anomalous position
                else:
                    lean_angle = 0
            
            # Only LIDAR 2 (lower side) active
            elif lidar1_point is None and lidar2_point is not None:
                # Get position information
                y_pos = lidar2_point[1]
                
                # If leaning down and on lower part: likely toward LIDAR 2 (negative lean)
                if side_lean_angle < 80 and y_pos < 0:
                    lean_angle = -side_lean_deviation * 0.5
                # If leaning down and on upper part: likely away from LIDAR 1 (negative lean)
                elif side_lean_angle < 80 and y_pos >= 0:
                    lean_angle = -side_lean_deviation * 0.3
                # Near vertical or anomalous position
                else:
                    lean_angle = 0
            
            # No LIDARs (shouldn't happen in this function but handle it)
            else:
                lean_angle = 0
                
            # Use a scaled confidence based on side lean deviation
            confidence = min(0.8, side_lean_deviation / 90)
            
        # Case 3: No valid data
        else:
            lean_angle = 0
            confidence = 0
        
        # Clamp to reasonable range
        lean_angle = max(-self.MAX_UP_DOWN_LEAN, min(self.MAX_UP_DOWN_LEAN, lean_angle))
        
        return lean_angle, confidence

    def project_lidar_point_with_3d_lean(self, lidar_point, lidar_height, side_lean_angle, up_down_lean_angle, camera_y):
        """
        Project a LIDAR detection point to account for both side-to-side and up/down lean.
        
        Args:
            lidar_point: (x, y) position of LIDAR detection
            lidar_height: height of the LIDAR beam above board in mm
            side_lean_angle: angle of dart from vertical in degrees (90° = vertical, from camera)
            up_down_lean_angle: angle of up/down lean in degrees (0° = vertical)
            camera_y: Y-coordinate from camera detection
            
        Returns:
            (x, y) position of the adjusted point
        """
        if lidar_point is None:
            return lidar_point
            
        # Handle missing lean angles
        if side_lean_angle is None:
            side_lean_angle = 90  # Default to vertical
        if up_down_lean_angle is None:
            up_down_lean_angle = 0  # Default to vertical
        
        # Extract original coordinates
        original_x, original_y = lidar_point
        
        # Apply side-to-side lean adjustment if we have camera data
        adjusted_y = original_y
        if camera_y is not None:
            # Calculate side-to-side lean adjustment (similar to your existing code)
            # Convert to 0-1 scale where 0 is horizontal (0°) and 1 is vertical (90°)
            side_lean_factor = side_lean_angle / 90.0
            inverse_side_lean = 1.0 - side_lean_factor
            
            # Calculate Y displacement (how far LIDAR point is from camera line)
            y_displacement = original_y - camera_y
            
            # Apply side-to-side adjustment proportional to lean angle, with constraints
            MAX_SIDE_ADJUSTMENT = self.side_lean_max_adjustment  # mm
            side_adjustment = min(inverse_side_lean * abs(y_displacement), MAX_SIDE_ADJUSTMENT)
            side_adjustment *= -1 if y_displacement > 0 else 1
            
            # Apply side-to-side adjustment
            adjusted_y = original_y + side_adjustment
        
        # Calculate up/down adjustment
        # This is proportional to the up/down lean angle and the distance from board center
        # The farther from center, the more adjustment needed for the same lean angle
        
        # Distance from board center in Y direction
        y_distance_from_center = abs(original_y)
        
        # Calculate up/down adjustment 
        # Positive up_down_lean_angle means leaning upward, adjustment should be in X direction
        MAX_UP_DOWN_ADJUSTMENT = self.up_down_lean_max_adjustment  # mm
        
        # Calculate adjustment proportional to lean angle and distance from center
        # More lean and further from center = bigger adjustment
        up_down_adjustment = (up_down_lean_angle / 30.0) * (y_distance_from_center / 170.0) * MAX_UP_DOWN_ADJUSTMENT
        
        # Apply up/down adjustment to X coordinate
        adjusted_x = original_x + up_down_adjustment
        
        return (adjusted_x, adjusted_y)

    def find_camera_board_intersection(self, camera_y):
        """Calculate where the camera epipolar line intersects the board surface.
        Camera is now on the left, so epipolar line is horizontal.
        
        Args:
            camera_y: Y position detected by camera in mm
            
        Returns:
            (x, y) position of intersection point on board surface
        """
        if camera_y is None:
            return None
            
        # Simple case - the Y coordinate is directly from camera, X is 0 (board surface)
        # In reality, we might need a more complex projection based on camera angle
        return (0, camera_y)

    def calculate_final_tip_position(self, camera_point, lidar1_point, lidar2_point):
        """
        Calculate the final tip position using all available data with enhanced 3D lean correction.
        Updated for camera on the left and up/down lean instead of forward/backward.
        
        Args:
            camera_point: Intersection of camera vector with board
            lidar1_point: Projected LIDAR 1 point
            lidar2_point: Projected LIDAR 2 point
            
        Returns:
            (x, y) final estimated tip position
        """
        # Points that are actually available
        valid_points = []
        if camera_point is not None:
            valid_points.append(camera_point)
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
            
        # Define maximum allowed discrepancy between sensors
        MAX_DISCREPANCY = 15.0  # mm
        
        # Enhanced weighting system that considers up/down lean
        if camera_point is not None:
            if lidar1_point is not None and lidar2_point is not None:
                # Detect up/down lean
                up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
                
                # Significant lean detected with good confidence
                if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
                    # Direction of lean affects which LIDAR to trust more for X position
                    if up_down_lean_angle > 0:  # Leaning upward
                        # Give more weight to LIDAR 1 for X position
                        lidar_x = lidar1_point[0] * 0.7 + lidar2_point[0] * 0.3
                    else:  # Leaning downward
                        # Give more weight to LIDAR 2 for X position
                        lidar_x = lidar1_point[0] * 0.3 + lidar2_point[0] * 0.7
                else:
                    # No significant lean detected, use average of both LIDARs
                    lidar_x = (lidar1_point[0] + lidar2_point[0]) / 2
                    
                # Use camera for Y position (more reliable since camera is now on the left)
                final_x = lidar_x
                final_y = camera_point[1]
                
                final_tip_position = (final_x, final_y)
            elif lidar1_point is not None:
                # Have camera and LIDAR 1
                final_tip_position = (lidar1_point[0], camera_point[1])
            elif lidar2_point is not None:
                # Have camera and LIDAR 2
                final_tip_position = (lidar2_point[0], camera_point[1])
            else:
                final_tip_position = camera_point
        
        # If only LIDARs are available (no camera)
        elif lidar1_point is not None and lidar2_point is not None:
            # Detect up/down lean to adjust weighting
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
            
            # If significant lean with good confidence
            if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
                if up_down_lean_angle > 0:  # Leaning upward
                    # Weight LIDAR 1 more for both X and Y
                    weight1 = 0.7
                    weight2 = 0.3
                else:  # Leaning downward
                    # Weight LIDAR 2 more for both X and Y
                    weight1 = 0.3
                    weight2 = 0.7
                    
                final_x = lidar1_point[0] * weight1 + lidar2_point[0] * weight2
                final_y = lidar1_point[1] * weight1 + lidar2_point[1] * weight2
            else:
                # No significant lean, average the positions
                final_x = (lidar1_point[0] + lidar2_point[0]) / 2
                final_y = (lidar1_point[1] + lidar2_point[1]) / 2
                
            final_tip_position = (final_x, final_y)
        else:
            # This shouldn't happen if the earlier logic is correct
            final_tip_position = valid_points[0]
        
        # Apply scale correction to final position
        if final_tip_position is not None:
            x, y = final_tip_position
            x = x * self.x_scale_correction
            y = y * self.y_scale_correction
            final_tip_position = (x, y)
            
        return final_tip_position

    def update_lean_visualization(self, side_lean_angle, up_down_lean_angle, lean_confidence):
        """Update the visualization of lean angles."""
        # Handle None values for lean angles
        if side_lean_angle is None:
            side_lean_angle = 90.0  # Default to vertical
        if up_down_lean_angle is None:
            up_down_lean_angle = 0.0
        if lean_confidence is None:
            lean_confidence = 0.0
            
        # Update text for lean angles
        self.lean_text.set_text(
            f"Side Lean: {side_lean_angle:.1f}° (90° = vertical)\n"
            f"Up/Down Lean: {up_down_lean_angle:.1f}° (conf: {lean_confidence:.2f})"
        )
        
        # If we have a good up/down lean detection, visualize it with an arrow
        if lean_confidence > 0.6 and abs(up_down_lean_angle) > 5:
            # Create an arrow showing the up/down lean direction
            # Arrow starts at origin (0,0)
            arrow_length = 50  # Length of arrow
            
            # Arrow direction depends on up/down lean angle
            # Positive angle means leaning upward
            # Negative angle means leaning downward
            
            # Calculate arrow endpoint
            # If leaning upward, arrow points right and up
            # If leaning downward, arrow points right and down
            if up_down_lean_angle > 0:
                # Leaning upward
                arrow_dx = arrow_length * np.cos(np.radians(up_down_lean_angle))
                arrow_dy = arrow_length * np.sin(np.radians(up_down_lean_angle))
            else:
                # Leaning downward
                arrow_dx = arrow_length * np.cos(np.radians(-up_down_lean_angle))
                arrow_dy = -arrow_length * np.sin(np.radians(-up_down_lean_angle))
                
            # Add or update arrow annotation
            if hasattr(self, 'lean_arrow') and self.lean_arrow is not None:
                self.lean_arrow.remove()
            self.lean_arrow = self.ax.arrow(
                0, 0, arrow_dx, arrow_dy, 
                width=3, head_width=10, head_length=10, 
                fc='purple', ec='purple', alpha=0.7
            )
            
            # Add text label near arrow
            if hasattr(self, 'arrow_text') and self.arrow_text is not None:
                self.arrow_text.remove()
            self.arrow_text = self.ax.text(
                arrow_dx/2, arrow_dy/2, 
                f"{abs(up_down_lean_angle):.1f}°", 
                color='purple', fontsize=9, 
                ha='center', va='center'
            )
        else:
            # Remove arrow if lean not detected with confidence
            if hasattr(self, 'lean_arrow') and self.lean_arrow is not None:
                self.lean_arrow.remove()
                self.lean_arrow = None
            if hasattr(self, 'arrow_text') and self.arrow_text is not None:
                self.arrow_text.remove()
                self.arrow_text = None
            
    def update_plot(self, frame):
        """Update plot data with enhanced 3D lean correction and CSV logging."""
        x1, y1 = [], []
        x2, y2 = [], []
        
        # Track the most significant LIDAR points for vector visualization
        lidar1_most_significant = None
        lidar2_most_significant = None
        
        # Process LIDAR 1 data
        while not self.lidar1_queue.empty():
            angle, distance = self.lidar1_queue.get()
            x, y = self.polar_to_cartesian(angle, distance - self.lidar1_offset, 
                                         self.lidar1_pos, self.lidar1_rotation, self.lidar1_mirror)
            
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
                    if lidar1_most_significant is None or dist_from_center < lidar1_most_significant[2]:
                        lidar1_most_significant = (x, y, dist_from_center)
                        
        # Process LIDAR 2 data
        while not self.lidar2_queue.empty():
            angle, distance = self.lidar2_queue.get()
            x, y = self.polar_to_cartesian(angle, distance - self.lidar2_offset, 
                                         self.lidar2_pos, self.lidar2_rotation, self.lidar2_mirror)
            
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
                    if lidar2_most_significant is None or dist_from_center < lidar2_most_significant[2]:
                        lidar2_most_significant = (x, y, dist_from_center)

        # Calculate LIDAR average positions if enough data
        lidar1_avg = None
        lidar2_avg = None
        
        if len(self.lidar1_recent_points) > 0:
            avg_x = sum(p[0] for p in self.lidar1_recent_points) / len(self.lidar1_recent_points)
            avg_y = sum(p[1] for p in self.lidar1_recent_points) / len(self.lidar1_recent_points)
            lidar1_avg = (avg_x, avg_y)
            
        if len(self.lidar2_recent_points) > 0:
            avg_x = sum(p[0] for p in self.lidar2_recent_points) / len(self.lidar2_recent_points)
            avg_y = sum(p[1] for p in self.lidar2_recent_points) / len(self.lidar2_recent_points)
            lidar2_avg = (avg_x, avg_y)
        
        # Get camera data and side-to-side lean angle
        camera_y = self.camera_data.get("dart_mm_y")
        side_lean_angle = self.camera_data.get("dart_angle", 90)  # Default to vertical if unknown
        
        # Calculate up/down lean angle using both LIDARs if available
        up_down_lean_angle = 0
        lean_confidence = 0
        if lidar1_avg is not None and lidar2_avg is not None:
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_avg, lidar2_avg)
            
            # Add to lean history for smoothing
            self.lean_history.append((up_down_lean_angle, lean_confidence))
            if len(self.lean_history) > self.max_lean_history:
                self.lean_history.pop(0)
                
            # Calculate weighted average of lean angles (higher confidence = higher weight)
            if self.lean_history:
                total_weight = sum(conf for _, conf in self.lean_history)
                if total_weight > 0:
                    smoothed_lean = sum(angle * conf for angle, conf in self.lean_history) / total_weight
                    up_down_lean_angle = smoothed_lean
                    
            # Update current values for visualization
            self.current_up_down_lean_angle = up_down_lean_angle
            self.up_down_lean_confidence = lean_confidence
        
        # Calculate intersection of camera vector with board surface
        self.camera_board_intersection = self.find_camera_board_intersection(camera_y)
        
        # Project LIDAR points accounting for both side-to-side and up/down lean
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None
        
        if lidar1_avg is not None:
            self.lidar1_projected_point = self.project_lidar_point_with_3d_lean(
                lidar1_avg, self.lidar1_height, side_lean_angle, up_down_lean_angle, camera_y)
                
        if lidar2_avg is not None:
            self.lidar2_projected_point = self.project_lidar_point_with_3d_lean(
                lidar2_avg, self.lidar2_height, side_lean_angle, up_down_lean_angle, camera_y)
        
        # If camera has no data, don't adjust LIDAR points for side-to-side lean
        # but still adjust for up/down lean if detected
        if camera_y is None and lidar1_avg is not None and lidar2_avg is not None:
            self.lidar1_projected_point = self.project_lidar_point_with_3d_lean(
                lidar1_avg, self.lidar1_height, 90, up_down_lean_angle, None)
            self.lidar2_projected_point = self.project_lidar_point_with_3d_lean(
                lidar2_avg, self.lidar2_height, 90, up_down_lean_angle, None)
        
        # Calculate final tip position using all available data with enhanced 3D lean correction
        final_tip_position = self.calculate_final_tip_position(
            self.camera_board_intersection, 
            self.lidar1_projected_point,
            self.lidar2_projected_point
        )
        
        # Apply segment coefficients to adjust for wire proximity effects
        if final_tip_position is not None:
            final_tip_position = self.apply_segment_coefficients(*final_tip_position)

        # Update scatter plots with LIDAR data
        self.scatter1.set_data(x1, y1)
        self.scatter2.set_data(x2, y2)
        
        # Update vectors if we have most significant points
        if lidar1_most_significant is not None:
            x, y, _ = lidar1_most_significant
            self.lidar1_vector.set_data([self.lidar1_pos[0], x], [self.lidar1_pos[1], y])
        
        if lidar2_most_significant is not None:
            x, y, _ = lidar2_most_significant
            self.lidar2_vector.set_data([self.lidar2_pos[0], x], [self.lidar2_pos[1], y])
        
        # Update camera vector if we have a valid intersection
        if self.camera_board_intersection is not None:
            self.camera_vector.set_data(
                [self.camera_position[0], self.camera_board_intersection[0]], 
                [self.camera_position[1], self.camera_board_intersection[1]]
            )
            self.camera_dart.set_data(
                [self.camera_board_intersection[0]], 
                [self.camera_board_intersection[1]]
            )
        
        # Update LIDAR projected points
        if self.lidar1_projected_point is not None:
            self.lidar1_dart.set_data([self.lidar1_projected_point[0]], [self.lidar1_projected_point[1]])
        
        if self.lidar2_projected_point is not None:
            self.lidar2_dart.set_data([self.lidar2_projected_point[0]], [self.lidar2_projected_point[1]])
        
        # Update final tip position
        if final_tip_position is not None:
            self.detected_dart.set_data([final_tip_position[0]], [final_tip_position[1]])
        
        # Update lean visualization
        self.update_lean_visualization(side_lean_angle, up_down_lean_angle, lean_confidence)
        
        # Get tip pixel data from camera if available
        tip_pixel = self.camera_data.get("tip_pixel")
        
        # Log dart data to CSV
        self.log_dart_data(
            final_tip_position,
            tip_pixel,
            side_lean_angle,
            self.current_up_down_lean_angle
        )
        
        return self.scatter1, self.scatter2, self.camera_vector, self.lidar1_vector, self.lidar2_vector, \
               self.camera_dart, self.lidar1_dart, self.lidar2_dart, self.detected_dart

    def start(self):
        """Start all threads and visualization."""
        # Start background threads
        lidar1_thread = threading.Thread(
            target=self.start_lidar, 
            args=("./lidar_reader.py --port /dev/ttyUSB0", self.lidar1_queue, 1),
            daemon=True
        )
        lidar2_thread = threading.Thread(
            target=self.start_lidar, 
            args=("./lidar_reader.py --port /dev/ttyUSB1", self.lidar2_queue, 2),
            daemon=True
        )
        camera_thread = threading.Thread(target=self.camera_detection, daemon=True)
        
        lidar1_thread.start()
        lidar2_thread.start()
        camera_thread.start()
        
        # Start animation
        self.ani = FuncAnimation(
            self.fig, self.update_plot, 
            blit=True, interval=100, 
            cache_frame_data=False
        )
        
        plt.show()

if __name__ == "__main__":
    visualizer = LidarCameraVisualizer()
    visualizer.start()
