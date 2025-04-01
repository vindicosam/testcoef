#!/usr/bin/env python3
"""
print2.py – Integrated Dart Tracking Visualizer

This script uses a single camera (positioned to the left of the board)
to detect a dart tip. The camera detection uses a pixel‐to‑mm calibration 
(interpolated from provided calibration points) to convert the detected pixel 
to a board y-coordinate. Because a single camera can only define a ray, we 
project an 800 mm vector from the camera along a fixed direction (assumed to be horizontal)
to yield a candidate tip position. Since the camera is on the left, a detected 
lean is applied as a y‑axis adjustment: a left lean moves the tip downward and a 
right lean moves it upward.

Additionally, LIDAR threads are started (stubbed here) to allow fusion with 
other sensor data (if available).

Adjust parameters as needed for your specific setup.
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import csv
import time
import os
from datetime import datetime
from queue import Queue
import threading
import subprocess
import signal
import sys
import json

class LidarCameraVisualizer:
    def __init__(self):
        # -------------------------------
        # LIDAR Setup (if available – stubbed for now)
        # -------------------------------
        self.lidar1_pos = (-202.5, 224.0)  # in mm relative to board center
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
        self.camera_vector_length = 800    # We project an 800mm vector from the camera
        # ROI for the camera (values in pixels)
        self.camera_board_plane_y = 250   # y position (in pixels) where board surface lies
        self.camera_roi_range = 30
        self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
        self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
        self.camera_roi_left = 119
        self.camera_roi_right = 604

        # Calibration points for converting a pixel x-coordinate to board mm y-coordinate.
        # (Each tuple is (pixel_x, mm_y). Adjust these based on your calibration.)
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
        # Fallback linear conversion parameters (if not enough points)
        self.pixel_to_mm_factor = -0.628
        self.pixel_offset = 192.8

        # Detection persistence parameters
        self.last_valid_detection = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30

        # Background subtraction & frame differencing parameters (from previous alean.py)
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=162, varThreshold=67, detectShadows=False
        )
        self.prev_gray = None
        self.diff_threshold = 25
        self.min_contour_area = 30

        # Variables to hold the camera detection result
        # camera_data will store:
        #   "dart_mm_y": the board y coordinate computed from calibration,
        #   "dart_angle": the measured lean angle (90° means vertical),
        #   "tip_pixel": the detected tip pixel (global) coordinate.
        self.camera_data = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.current_cam_lean_angle = None  # in degrees
        self.current_cam_lean_direction = "VERTICAL"  # "LEFT", "RIGHT", or "VERTICAL"

        # Lean offset parameters – note that since the camera is on the left,
        # we will adjust the final tip position along the y axis.
        self.side_lean_max_adjustment = 6.0   # Maximum y-axis adjustment in mm

        # -------------------------------
        # Dartboard Visualization and Scoring
        # -------------------------------
        self.board_scale_factor = 2.75
        # Load the dartboard image (ensure this file is present)
        self.dartboard_image = mpimg.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        # Define dartboard region radii (in mm) for scoring
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

        # Running flag and signal handling
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)

        # Set up visualization plot
        self.setup_plot()

    # -------------------------------
    # Pixel-to-mm conversion (using calibration interpolation)
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
    # Measure dart tip angle using a RANSAC‐style method (adapted from alean.py)
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
        # Determine lean direction: if angle <85 then LEFT, if >95 then RIGHT, else VERTICAL
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
    # Camera detection – using background subtraction and frame differencing
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
            # For a left camera, rotate frame 180°
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

            # Reset current detection data
            self.camera_data["dart_mm_y"] = None
            self.camera_data["dart_angle"] = None
            self.camera_data["tip_pixel"] = None

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tip_point = None
            if contours:
                # For a left camera, choose the rightmost contour (largest x)
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
                # Convert tip coordinate in ROI to global pixel coordinate
                global_pixel_x = tip_point[0] + self.camera_roi_left
                global_pixel_y = tip_point[1] + self.camera_roi_top
                # Use the calibration conversion to get board y-coordinate
                dart_mm_y = self.pixel_to_mm(global_pixel_x)
                self.camera_data["dart_mm_y"] = dart_mm_y
                self.camera_data["tip_pixel"] = (global_pixel_x, global_pixel_y)
                self.last_valid_detection = self.camera_data.copy()
                self.detection_persistence_counter = self.detection_persistence_frames
            elif self.detection_persistence_counter > 0:
                self.detection_persistence_counter -= 1
                if self.detection_persistence_counter > 0:
                    self.camera_data = self.last_valid_detection.copy()

            # For debugging – show ROI and mask
            cv2.imshow("Camera ROI", roi)
            cv2.imshow("Combined Mask", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------
    # Compute the camera vector based on the detection
    # -------------------------------
    def compute_camera_vector(self):
        """
        Since a single camera only provides a ray,
        we define the final camera-determined tip as the point
        that lies 800mm from the camera along a horizontal line,
        with the board y-coordinate taken from the calibration conversion.
        """
        if self.camera_data.get("dart_mm_y") is None or self.camera_data.get("tip_pixel") is None:
            return None
        # We assume the horizontal (x) component of the detected tip comes solely from the fixed vector.
        # Thus, the final camera vector point is:
        #   (camera_position_x + 800, dart_mm_y)
        cam_x = self.camera_position[0]
        final_x = cam_x + self.camera_vector_length
        final_y = self.camera_data["dart_mm_y"]
        return (final_x, final_y)

    # -------------------------------
    # Apply lean offset adjustment on the y axis.
    # For a camera on the left, a left lean (angle < 90) moves the tip downward,
    # and a right lean (angle > 90) moves it upward.
    # -------------------------------
    def apply_lean_offset(self, point):
        if point is None or self.current_cam_lean_angle is None:
            return point
        x, y = point
        deviation = 90 - self.current_cam_lean_angle  # positive if lean to LEFT, negative if RIGHT
        # Calculate offset proportionally (max offset when deviation is 90°)
        offset = (deviation / 90.0) * self.side_lean_max_adjustment
        # For a left camera, a positive deviation (left lean) moves the tip downward (decrease y)
        adjusted_y = y - offset
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
        self.ax.set_title("Dartboard Visualization")
        self.ax.grid(True)
        self.update_dartboard_image()
        # Plot fixed sensor positions
        self.ax.plot(*self.camera_position, "ro", label="Camera")
        self.ax.plot(*self.lidar1_pos, "bo", label="LIDAR 1")
        self.ax.plot(*self.lidar2_pos, "go", label="LIDAR 2")
        # Markers for camera vector and final tip position
        self.camera_vector_marker, = self.ax.plot([], [], "r--", label="Camera Vector")
        self.detected_tip_marker, = self.ax.plot([], [], "rx", markersize=10, label="Final Tip")
        self.lean_text = self.ax.text(-380, 380, "", fontsize=9)
        self.ax.legend(loc="upper right", fontsize=8)

    def update_dartboard_image(self):
        scaled_extent = [-170 * self.board_scale_factor, 170 * self.board_scale_factor,
                         -170 * self.board_scale_factor, 170 * self.board_scale_factor]
        self.ax.imshow(self.dartboard_image, extent=scaled_extent, zorder=0)

    def update_plot(self, frame):
        # Process camera data to compute the camera vector
        cam_vector = self.compute_camera_vector()  # This is (camera_position_x+800, dart_mm_y)
        # Apply lean offset on the y axis
        adjusted_cam_vector = self.apply_lean_offset(cam_vector)
        
        # Log data if available
        if adjusted_cam_vector is not None and self.camera_data.get("tip_pixel") is not None:
            self.log_dart_data(
                adjusted_cam_vector,
                self.camera_data["tip_pixel"],
                self.current_cam_lean_angle if self.current_cam_lean_angle is not None else 0,
                self.current_cam_lean_direction
            )
            # Update camera vector visualization (draw a line from camera position to the computed point)
            self.camera_vector_marker.set_data(
                [self.camera_position[0], adjusted_cam_vector[0]],
                [self.camera_position[1], adjusted_cam_vector[1]]
            )
            # Update final tip marker
            self.detected_tip_marker.set_data(
                [adjusted_cam_vector[0]], [adjusted_cam_vector[1]]
            )
        else:
            self.camera_vector_marker.set_data([], [])
            self.detected_tip_marker.set_data([], [])
        
        # Update lean text
        side_str = f"{self.current_cam_lean_angle:.1f}°" if self.current_cam_lean_angle is not None else "N/A"
        lean_text = f"Lean: {side_str} ({self.current_cam_lean_direction})"
        self.lean_text.set_text(lean_text)
        
        return [self.camera_vector_marker, self.detected_tip_marker, self.lean_text]

    # -------------------------------
    # LIDAR reading thread (stubbed – adjust script paths as needed)
    # -------------------------------
    def start_lidar(self, script_path, queue_obj, lidar_id):
        try:
            process = subprocess.Popen([script_path], stdout=subprocess.PIPE, text=True)
            print(f"LIDAR {lidar_id} started.")
            while self.running:
                line = process.stdout.readline()
                if not line:
                    continue
                if "a:" in line and "d:" in line:
                    try:
                        parts = line.strip().split()
                        angle = float(parts[1].replace("a:", ""))
                        distance = float(parts[2].replace("d:", ""))
                        queue_obj.put((angle, distance))
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error starting LIDAR {lidar_id}: {e}")

    # -------------------------------
    # Main run loop
    # -------------------------------
    def run(self, lidar1_script, lidar2_script):
        # Start LIDAR threads (if available)
        lidar1_thread = threading.Thread(
            target=self.start_lidar,
            args=(lidar1_script, self.lidar1_queue, 1),
            daemon=True
        )
        lidar2_thread = threading.Thread(
            target=self.start_lidar,
            args=(lidar2_script, self.lidar2_queue, 2),
            daemon=True
        )
        camera_thread = threading.Thread(target=self.camera_detection, daemon=True)
        
        lidar1_thread.start()
        time.sleep(1)
        lidar2_thread.start()
        time.sleep(1)
        camera_thread.start()
        
        # Start matplotlib animation for visualization
        self.ani = FuncAnimation(
            self.fig, self.update_plot, blit=True, interval=100, cache_frame_data=False
        )
        plt.show()

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

if __name__ == "__main__":
    # Replace these with your actual LIDAR script paths or stubs.
    lidar1_script = "./lidar1_script.py"
    lidar2_script = "./lidar2_script.py"
    
    visualizer = LidarCameraVisualizer()
    # If no calibration mode is desired, simply run
    visualizer.run(lidar1_script, lidar2_script)
