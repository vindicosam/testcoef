#!/usr/bin/env python3
"""
print2.py – Integrated Dart Tracking Visualizer

This script uses a single camera (positioned to the left of the board)
to detect a dart tip. The camera detection uses a pixel‐to‐mm calibration 
(interpolated from provided calibration points) to convert the detected pixel 
to a board y-coordinate. Because a single camera can only define a ray, we 
project an 800 mm vector from the camera along a fixed horizontal direction 
to yield a candidate tip position. Since the camera is on the left, a detected 
lean is applied as a y‐axis adjustment: a left lean moves the tip downward and a 
right lean moves it upward.

Additionally, stub LIDAR threads are started (if available) so that later you 
can fuse LIDAR data with camera data.

Adjust parameters as needed for your specific setup.
"""

# Use TkAgg to avoid Qt/Wayland issues
import matplotlib
matplotlib.use("TkAgg")
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
        # -------------------------------
        # LIDAR Setup (if available – stubbed here)
        # -------------------------------
        self.lidar1_pos = (-202.5, 224.0)  # mm relative to board center
        self.lidar2_pos = (204.0, 223.5)
        self.lidar1_rotation = 342.5
        self.lidar2_rotation = 186.25
        self.lidar1_mirror = True
        self.lidar2_mirror = True
        self.lidar1_queue = Queue()
        self.lidar2_queue = Queue()
        self.lidar1_recent_points = []
        self.lidar2_recent_points = []
        self.max_recent_points = 20

        # -------------------------------
        # Camera Setup (camera is on the left)
        # -------------------------------
        self.camera_position = (-350, 0)   # in board mm (fixed position)
        self.camera_vector_length = 800    # We project an 800 mm vector from the camera
        # ROI for the camera (in pixels)
        self.camera_board_plane_y = 250    # pixel y where board surface lies
        self.camera_roi_range = 30
        self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
        self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
        self.camera_roi_left = 119
        self.camera_roi_right = 604

        # Calibration points for converting pixel x to board mm y.
        # (Each tuple: (pixel_x, mm_y))
        self.camera_calibration_points = [
            (151, 50),
            (277, -100),
            (290, 0),
            (359, -50),
            (327, 103),
            (506, 0),
            (68, 0),
            (290, 171),
            (290, -171)
        ]
        self.camera_calibration_points.sort(key=lambda p: p[0])
        # Fallback conversion parameters
        self.pixel_to_mm_factor = -0.628
        self.pixel_offset = 192.8

        # Detection persistence parameters
        self.last_valid_detection = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30

        # Background subtractor & frame differencing parameters
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=162, varThreshold=67, detectShadows=False
        )
        self.prev_gray = None
        self.diff_threshold = 25
        self.min_contour_area = 30

        # Camera detection result storage:
        # "dart_mm_y": computed board y coordinate,
        # "dart_angle": measured lean angle (90° is vertical),
        # "tip_pixel": detected tip pixel (global) coordinate.
        self.camera_data = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.current_cam_lean_angle = None  # in degrees
        self.current_cam_lean_direction = "VERTICAL"  # "LEFT", "RIGHT", or "VERTICAL"

        # Lean offset: for a left camera, a left lean will decrease the final y value.
        self.side_lean_max_adjustment = 6.0  # maximum y-axis adjustment in mm

        # -------------------------------
        # Dartboard Visualization and Scoring
        # -------------------------------
        self.board_scale_factor = 2.75
        # Load dartboard image (ensure the file exists in the same directory)
        self.dartboard_image = mpimg.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        # Define board region radii in mm (for scoring, etc.)
        self.radii = {
            "bullseye": 6.35,
            "outer_bull": 15.9,
            "inner_treble": 99,
            "outer_treble": 107,
            "inner_double": 162,
            "outer_double": 170,
            "board_edge": 195,
        }

        # CSV logging initialization
        self.initialize_csv_logging()

        # Additional data for LIDAR fusion (if available)
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None
        self.camera_board_intersection = None

        # For displaying score text on the plot
        self.score_text = None

        # Additional calibration correction (stubbed here)
        self.calibration_points = {}  # you can add your own calibration corrections if desired

        # Scale corrections for board dimensions (if needed)
        self.x_scale_correction = 1.02
        self.y_scale_correction = 1.04

        # 3D lean detection variables (stubbed – not used by a single camera)
        self.current_up_down_lean_angle = 0.0
        self.up_down_lean_confidence = 0.0
        self.lean_history = []
        self.max_lean_history = 60
        self.lean_arrow = None
        self.arrow_text = None
        self.MAX_SIDE_LEAN = 35.0
        self.MAX_UP_DOWN_LEAN = 30.0
        self.MAX_X_DIFF_FOR_MAX_LEAN = 4.0

        # NEW: Per-segment radial offsets (in mm)
        self.segment_radial_offsets = {}
        for segment in range(1, 21):
            self.segment_radial_offsets[segment] = -15

        # Coefficient dictionaries for ring corrections (stubbed; use defaults)
        self.large_segment_coeff = {}  # ... [omitted for brevity – see your working version]
        self.doubles_coeff = {}
        self.trebles_coeff = {}
        self.small_segment_coeff = {}

        # Coefficient strength scaling factors
        self.coefficient_scaling = {}
        for segment in range(1, 21):
            self.coefficient_scaling[segment] = {
                'doubles': 1.0,
                'trebles': 1.0,
                'small': 1.0,
                'large': 1.0
            }

        # Set running flag and signal handling
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)

        # Set up visualization plot
        self.setup_plot()

    # -------------------------------
    # Pixel-to-mm conversion using calibration points
    # -------------------------------
    def pixel_to_mm(self, pixel_x):
        if len(self.camera_calibration_points) >= 2:
            for i in range(len(self.camera_calibration_points) - 1):
                p1, mm1 = self.camera_calibration_points[i]
                p2, mm2 = self.camera_calibration_points[i+1]
                if p1 <= pixel_x <= p2:
                    return mm1 + (pixel_x - p1) * (mm2 - mm1) / (p2 - p1)
            if pixel_x < self.camera_calibration_points[0][0]:
                return self.camera_calibration_points[0][1]
            else:
                return self.camera_calibration_points[-1][1]
        else:
            return self.pixel_to_mm_factor * pixel_x + self.pixel_offset

    # -------------------------------
    # Measure dart tip angle using RANSAC‐style method
    # -------------------------------
    def measure_tip_angle(self, mask, tip_point):
        if tip_point is None:
            return None
        tip_x, tip_y = tip_point
        search_depth = 25
        search_width = 40
        min_points = 8
        min_x = max(0, tip_x - search_width)
        max_x = min(mask.shape[1] - 1, tip_x + search_width)
        max_y = min(mask.shape[0] - 1, tip_y + search_depth)
        points_below = []
        for y in range(tip_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if mask[y, x] > 0:
                    points_below.append((x, y))
        if len(points_below) < min_points:
            return None
        best_angle = None
        best_inliers = 0
        for _ in range(10):
            if len(points_below) < 2:
                continue
            indices = np.random.choice(len(points_below), 2, replace=False)
            p1 = points_below[indices[0]]
            p2 = points_below[indices[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if math.hypot(dx, dy) < 5:
                continue
            if dx == 0:
                angle = 90
            else:
                slope = dy / dx
                angle_from_horizontal = math.degrees(math.atan(slope))
                angle = 90 - angle_from_horizontal
            inliers = []
            for point in points_below:
                if dx == 0:
                    dist_to_line = abs(point[0] - p1[0])
                else:
                    a = -slope
                    b = 1
                    c = slope * p1[0] - p1[1]
                    dist_to_line = abs(a * point[0] + b * point[1] + c) / math.sqrt(a*a + b*b)
                if dist_to_line < 2:
                    inliers.append(point)
            if len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_angle = angle
        if best_angle is None:
            pts = np.array(points_below)
            x_vals = pts[:, 0]
            y_vals = pts[:, 1]
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
            denominator = np.sum((x_vals - x_mean)**2)
            if denominator == 0:
                best_angle = 90
            else:
                slope = numerator / denominator
                best_angle = 90 - math.degrees(math.atan(slope))
        lean = "VERTICAL"
        if best_angle < 85:
            lean = "LEFT"
        elif best_angle > 95:
            lean = "RIGHT"
        return best_angle, lean, points_below

    # -------------------------------
    # Temporal filtering (moving median)
    # -------------------------------
    def apply_temporal_filtering(self, value, buffer):
        if value is None:
            return None
        buffer.append(value)
        if isinstance(value, tuple) and len(value) == 2:
            xs = sorted([p[0] for p in buffer if p is not None])
            ys = sorted([p[1] for p in buffer if p is not None])
            if xs and ys:
                return (xs[len(xs)//2], ys[len(ys)//2])
            else:
                return None
        else:
            vals = sorted([v for v in buffer if v is not None])
            if vals:
                return vals[len(vals)//2]
            else:
                return None

    # -------------------------------
    # Camera detection using background subtraction and frame differencing
    # -------------------------------
    def camera_detection(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1)
        self.prev_gray = None

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            # Rotate frame 180° for a left camera
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            roi = frame[self.camera_roi_top:self.camera_roi_bottom, self.camera_roi_left:self.camera_roi_right]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None or self.prev_gray.shape != gray.shape:
                self.prev_gray = gray.copy()
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            _, diff_thresh = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            self.prev_gray = gray.copy()
            fg_mask = self.camera_bg_subtractor.apply(gray)
            fg_mask = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY)[1]
            combined_mask = cv2.bitwise_or(fg_mask, diff_thresh)
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

            # Reset current detection
            self.camera_data["dart_mm_y"] = None
            self.camera_data["dart_angle"] = None
            self.camera_data["tip_pixel"] = None

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tip_point = None
            if contours:
                for contour in contours:
                    if cv2.contourArea(contour) > self.min_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        tip_point = (x + w//2, self.camera_board_plane_y - self.camera_roi_top)
                        break
            if tip_point is not None:
                angle_info = self.measure_tip_angle(combined_mask, tip_point)
                if angle_info is not None:
                    dart_angle, lean_dir, _ = angle_info
                    self.camera_data["dart_angle"] = dart_angle
                    self.current_cam_lean_angle = dart_angle
                    self.current_cam_lean_direction = lean_dir
                global_pixel_x = tip_point[0] + self.camera_roi_left
                global_pixel_y = tip_point[1] + self.camera_roi_top
                dart_mm_y = self.pixel_to_mm(global_pixel_x)
                self.camera_data["dart_mm_y"] = dart_mm_y
                self.camera_data["tip_pixel"] = (global_pixel_x, global_pixel_y)
                self.last_valid_detection = self.camera_data.copy()
                self.detection_persistence_counter = self.detection_persistence_frames
            elif self.detection_persistence_counter > 0:
                self.detection_persistence_counter -= 1
                if self.detection_persistence_counter > 0:
                    self.camera_data = self.last_valid_detection.copy()

            # Debug windows (can comment these out)
            cv2.imshow("Camera ROI", roi)
            cv2.imshow("Combined Mask", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------
    # Compute the camera vector (project an 800 mm ray)
    # -------------------------------
    def compute_camera_vector(self):
        if self.camera_data.get("dart_mm_y") is None or self.camera_data.get("tip_pixel") is None:
            return None
        cam_x = self.camera_position[0]
        final_x = cam_x + self.camera_vector_length
        final_y = self.camera_data["dart_mm_y"]
        return (final_x, final_y)

    # -------------------------------
    # Apply lean offset on the y axis (for a left camera)
    # -------------------------------
    def apply_lean_offset(self, point):
        if point is None or self.current_cam_lean_angle is None:
            return point
        x, y = point
        deviation = 90 - self.current_cam_lean_angle  # positive: left lean
        offset = (deviation / 90.0) * self.side_lean_max_adjustment
        adjusted_y = y - offset  # left lean moves tip downward
        return (x, adjusted_y)

    # -------------------------------
    # CSV Logging
    # -------------------------------
    def initialize_csv_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"dart_data_{timestamp}.csv"
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'CameraTip_X_mm', 'CameraTip_Y_mm', 
                                 'Tip_Pixel_X', 'Tip_Pixel_Y', 'Lean_Angle', 'Lean_Direction', 'Score'])
        print(f"CSV logging initialized: {self.csv_filename}")

    def log_dart_data(self, final_tip, tip_pixel, lean_angle, lean_direction):
        if final_tip is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        score = self.xy_to_dartboard_score(final_tip[0], final_tip[1])
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                timestamp,
                f"{final_tip[0]:.2f}",
                f"{final_tip[1]:.2f}",
                tip_pixel[0] if tip_pixel[0] is not None else "None",
                tip_pixel[1] if tip_pixel[1] is not None else "None",
                f"{lean_angle:.1f}" if lean_angle is not None else "None",
                lean_direction,
                score
            ])

    def xy_to_dartboard_score(self, x, y):
        dist = math.sqrt(x*x + y*y)
        if dist <= self.radii["bullseye"]:
            return "B"
        elif dist <= self.radii["outer_bull"]:
            return "OB"
        elif dist <= self.radii["outer_double"]:
            return "S"
        else:
            return "Outside"

    # -------------------------------
    # Visualization Setup and Update
    # -------------------------------
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_title("LIDAR and Camera Vector Visualization")
        self.ax.grid(True)
        self.update_dartboard_image()
        # Plot fixed sensor positions
        self.ax.plot(*self.camera_position, "ro", label="Camera")
        self.ax.plot(*self.lidar1_pos, "bo", label="LIDAR 1")
        self.ax.plot(*self.lidar2_pos, "go", label="LIDAR 2")
        # Markers for vectors and dart tip
        self.camera_vector, = self.ax.plot([], [], "r--", label="Camera Vector")
        self.camera_dart, = self.ax.plot([], [], "rx", markersize=8, label="Camera Intersection")
        self.lidar1_dart, = self.ax.plot([], [], "bx", markersize=8, label="LIDAR 1 Projected", zorder=3)
        self.lidar2_dart, = self.ax.plot([], [], "gx", markersize=8, label="LIDAR 2 Projected", zorder=3)
        self.detected_dart, = self.ax.plot([], [], "ro", markersize=4, label="Final Tip Position", zorder=10)
        self.scatter1, = self.ax.plot([], [], "b.", label="LIDAR 1 Data", zorder=3)
        self.scatter2, = self.ax.plot([], [], "g.", label="LIDAR 2 Data", zorder=3)
        self.lean_text = self.ax.text(-380, 380, "", fontsize=9)
        self.ax.legend(loc="upper right", fontsize=8)

    def update_dartboard_image(self):
        scaled_extent = [-170 * self.board_scale_factor, 170 * self.board_scale_factor,
                         -170 * self.board_scale_factor, 170 * self.board_scale_factor]
        self.ax.imshow(self.dartboard_image, extent=scaled_extent, zorder=0)

    # -------------------------------
    # LIDAR reading thread (stubbed – adjust script paths as needed)
    # -------------------------------
    def start_lidar(self, script_path, queue_obj, lidar_id):
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

    # -------------------------------
    # Main update function for animation
    # -------------------------------
    def update_plot(self, frame):
        # Process LIDAR data (if any)
        lidar1_points_x = []
        lidar1_points_y = []
        lidar2_points_x = []
        lidar2_points_y = []
        while not self.lidar1_queue.empty():
            angle, distance = self.lidar1_queue.get()
            x, y = self.polar_to_cartesian(angle, distance, self.lidar1_pos, self.lidar1_rotation, self.lidar1_mirror)
            if x is not None and y is not None:
                in_range, _ = self.filter_points_by_radii(x, y)
                if in_range:
                    lidar1_points_x.append(x)
                    lidar1_points_y.append(y)
                    self.lidar1_recent_points.append((x, y))
        while not self.lidar2_queue.empty():
            angle, distance = self.lidar2_queue.get()
            x, y = self.polar_to_cartesian(angle, distance, self.lidar2_pos, self.lidar2_rotation, self.lidar2_mirror)
            if x is not None and y is not None:
                in_range, _ = self.filter_points_by_radii(x, y)
                if in_range:
                    lidar2_points_x.append(x)
                    lidar2_points_y.append(y)
                    self.lidar2_recent_points.append((x, y))
        self.lidar1_recent_points = self.lidar1_recent_points[-self.max_recent_points:]
        self.lidar2_recent_points = self.lidar2_recent_points[-self.max_recent_points:]

        # Get camera data
        camera_y = self.camera_data["dart_mm_y"]
        side_lean_angle = self.camera_data["dart_angle"]
        # (For a single camera, up/down lean is not determined, so we set it to 0)
        up_down_lean_angle = 0
        lean_confidence = 0
        if len(self.lidar1_recent_points) > 0 and len(self.lidar2_recent_points) > 0:
            lidar1_point = self.lidar1_recent_points[-1]
            lidar2_point = self.lidar2_recent_points[-1]
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        self.update_lean_visualization(side_lean_angle, up_down_lean_angle, lean_confidence)
        camera_point = self.find_camera_board_intersection(camera_y)
        # (For this single camera mode, we rely solely on the camera vector)
        final_tip_position = self.apply_lean_offset(self.compute_camera_vector())
        # Log data to CSV
        self.log_dart_data(final_tip_position, self.camera_data["tip_pixel"],
                           side_lean_angle if side_lean_angle is not None else 0,
                           up_down_lean_angle)
        # Update plot markers
        self.scatter1.set_data(lidar1_points_x, lidar1_points_y)
        self.scatter2.set_data(lidar2_points_x, lidar2_points_y)
        if camera_point is not None:
            self.camera_vector.set_data([self.camera_position[0], camera_point[0]],
                                        [self.camera_position[1], camera_point[1]])
            self.camera_dart.set_data([camera_point[0]], [camera_point[1]])
        else:
            self.camera_vector.set_data([], [])
            self.camera_dart.set_data([], [])
        self.detected_dart.set_data([final_tip_position[0]], [final_tip_position[1]] if final_tip_position else ([], []))
        if side_lean_angle is not None:
            side_str = f"{side_lean_angle:.1f}°"
        else:
            side_str = "N/A"
        lean_text = f"Lean: {side_str} ({self.current_cam_lean_direction})"
        self.lean_text.set_text(lean_text)
        artists = [self.scatter1, self.scatter2, self.camera_vector, self.camera_dart,
                   self.detected_dart, self.lean_text]
        if hasattr(self, 'score_text') and self.score_text:
            artists.append(self.score_text)
        if hasattr(self, 'lean_arrow') and self.lean_arrow:
            artists.append(self.lean_arrow)
        if hasattr(self, 'arrow_text') and self.arrow_text:
            artists.append(self.arrow_text)
        return artists

    # -------------------------------
    # Main run loop: start LIDAR, camera, and animation threads
    # -------------------------------
    def run(self, lidar1_script, lidar2_script):
        lidar1_thread = threading.Thread(target=self.start_lidar,
                                         args=(lidar1_script, self.lidar1_queue, 1),
                                         daemon=True)
        lidar2_thread = threading.Thread(target=self.start_lidar,
                                         args=(lidar2_script, self.lidar2_queue, 2),
                                         daemon=True)
        camera_thread = threading.Thread(target=self.camera_detection, daemon=True)
        lidar1_thread.start()
        time.sleep(1)
        lidar2_thread.start()
        time.sleep(1)
        camera_thread.start()
        self.ani = FuncAnimation(self.fig, self.update_plot, blit=True, interval=100, cache_frame_data=False)
        plt.show()

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

    # --- (Calibration mode and LIDAR offset/scaling methods omitted for brevity) ---


if __name__ == "__main__":
    # Replace these script paths with the actual paths for your LIDAR scripts
    lidar1_script = "./tri_test_lidar1"
    lidar2_script = "./tri_test_lidar2"
    
    visualizer = LidarCameraVisualizer()
    
    # Optionally load coefficient scaling and segment offsets if available
    visualizer.load_coefficient_scaling()
    visualizer.load_segment_radial_offsets()
    
    # If no command line arguments, run normally
    visualizer.run(lidar1_script, lidar2_script)
