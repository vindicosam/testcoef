#!/usr/bin/env python3
"""
print2.py – Integrated Dart Tracking Visualizer

This script uses a single camera (positioned to the left of the board)
to detect a dart tip. It uses pixel‐to‐mm calibration (via interpolation)
to convert a detected pixel (x coordinate) into a board y-coordinate.
Since a single camera only defines a ray, we project an 800 mm vector from
the camera along a horizontal line. Then, based on the detected lean (from
the dart’s tip orientation), we adjust the projected tip’s y‑coordinate:
a left lean (angle < 90°) moves the tip downward while a right lean (angle > 90°)
moves it upward.

The script also starts LIDAR threads (if available) for sensor fusion,
provides extensive calibration routines, CSV logging, and visualization.
Adjust parameters as needed for your specific setup.
"""

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

        # -------------------------------
        # Background Subtraction & Frame Differencing (for camera detection)
        # -------------------------------
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=162, varThreshold=67, detectShadows=False
        )
        self.prev_gray = None
        self.diff_threshold = 25
        self.min_contour_area = 30

        # Variables to hold the camera detection result.
        # camera_data stores:
        #   "dart_mm_y": board y-coordinate computed from calibration,
        #   "dart_angle": measured lean angle (90° = vertical),
        #   "tip_pixel": detected tip (global pixel coordinates).
        self.camera_data = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.current_cam_lean_angle = None  # in degrees
        self.current_cam_lean_direction = "VERTICAL"  # "LEFT", "RIGHT", or "VERTICAL"

        # Lean offset parameters – for a left camera, we adjust the final tip's y-coordinate.
        self.side_lean_max_adjustment = 6.0   # Maximum y-axis adjustment in mm

        # -------------------------------
        # Dartboard Visualization and Scoring
        # -------------------------------
        self.board_scale_factor = 2.75
        self.dartboard_image = mpimg.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        self.radii = {
            "bullseye": 6.35,
            "outer_bull": 15.9,
            "inner_treble": 99,
            "outer_treble": 107,
            "inner_double": 162,
            "outer_double": 170,
            "board_edge": 195,
        }

        # -------------------------------
        # Calibration & Correction Data
        # -------------------------------
        self.calibration_points = {
            (0, 0): (-1.0, 0),           # Bullseye correction
            (23, 167): (0.9, 2.7),
            (-23, -167): (0.4, 0.6),
            (167, -24): (2.9, 2.6),
            (-167, 24): (-3.3, -2.7),
            (75, 75): (2.3, 2.1),
            (-75, -75): (2.2, -3.2),
        }
        self.x_scale_correction = 1.02
        self.y_scale_correction = 1.04

        # -------------------------------
        # 3D Lean Detection Variables
        # -------------------------------
        self.current_up_down_lean_angle = 0.0
        self.up_down_lean_confidence = 0.0
        self.lean_history = []
        self.max_lean_history = 60
        self.lean_arrow = None
        self.arrow_text = None
        self.MAX_SIDE_LEAN = 35.0
        self.MAX_UP_DOWN_LEAN = 30.0
        self.MAX_X_DIFF_FOR_MAX_LEAN = 4.0

        # -------------------------------
        # Segment-specific Calibration Data
        # -------------------------------
        self.segment_radial_offsets = {}
        for segment in range(1, 21):
            self.segment_radial_offsets[segment] = -15  # Default offset in mm

        # Coefficient dictionaries (for different scoring areas)
        # (These dictionaries have been carried over nearly verbatim from your original code.)
        self.large_segment_coeff = {
            "14_5": {"x_correction": -1.888, "y_correction": 12.790},
            "11_4": {"x_correction": -6.709, "y_correction": 14.045},
            # ... (all other keys as in your original code)
            "10_5": {"x_correction": -3.798, "y_correction": -0.596}
        }
        self.doubles_coeff = {
            "1_1": {"x_correction": 3.171, "y_correction": 0.025},
            "14_5": {"x_correction": 1.920, "y_correction": 6.191},
            # ... (all other keys as in your original code)
            "3_1": {"x_correction": -12.581, "y_correction": 8.704}
        }
        self.trebles_coeff = {
            "1_1": {"x_correction": 3.916, "y_correction": 7.238},
            "1_5": {"x_correction": 2.392, "y_correction": 0.678},
            # ... (all other keys as in your original code)
            "3_3": {"x_correction": -12.353, "y_correction": 8.187}
        }
        self.small_segment_coeff = {
            "8_5": {"x_correction": -7.021, "y_correction": 9.646},
            "5_1": {"x_correction": -2.830, "y_correction": 9.521},
            # ... (all other keys as in your original code)
            "4_5": {"x_correction": -2.475, "y_correction": 6.135}
        }

        # Initialize coefficient strength scaling for all segments
        self.coefficient_scaling = {}
        for segment in range(1, 21):
            self.coefficient_scaling[segment] = {
                'doubles': 1.0,
                'trebles': 1.0,
                'small': 1.0,
                'large': 1.0
            }

        # -------------------------------
        # Initialization Complete
        # -------------------------------
        self.initialize_csv_logging()
        self.score_text = None
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        self.setup_plot()

    # -------------------------------
    # Pixel-to-mm conversion via interpolation
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

    def clear_calibration_points(self):
        self.camera_calibration_points = []
        print("Calibration points cleared.")

    # -------------------------------
    # CSV Logging
    # -------------------------------
    def initialize_csv_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"dart_data_{timestamp}.csv"
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Dart_X_mm', 'Dart_Y_mm', 
                                 'Tip_Pixel_X', 'Tip_Pixel_Y', 
                                 'Side_Lean_Angle', 'Up_Down_Lean_Angle', 'Score'])
        print(f"CSV logging initialized: {self.csv_filename}")

    def log_dart_data(self, final_tip_position, tip_pixel, side_lean_angle, up_down_lean_angle):
        if final_tip_position is None:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        score = "None"
        if final_tip_position:
            score = self.xy_to_dartboard_score(final_tip_position[0], final_tip_position[1])
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                timestamp,
                f"{final_tip_position[0]:.2f}" if final_tip_position else "None",
                f"{final_tip_position[1]:.2f}" if final_tip_position else "None",
                f"{tip_pixel[0]}" if tip_pixel else "None",
                f"{tip_pixel[1]}" if tip_pixel else "None",
                f"{side_lean_angle:.2f}" if side_lean_angle is not None else "None",
                f"{up_down_lean_angle:.2f}" if up_down_lean_angle is not None else "None",
                score
            ])

    # -------------------------------
    # Dartboard Scoring
    # -------------------------------
    def xy_to_dartboard_score(self, x, y):
        distance = np.sqrt(x * x + y * y)
        angle = np.degrees(np.arctan2(y, x))
        angle = (angle - 9 + 360) % 360
        scores = [13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6]
        segment_idx = int(angle / 18)
        if segment_idx >= len(scores):
            return "Outside"
        base_score = scores[segment_idx]
        if distance <= self.radii["bullseye"]:
            return "B"
        elif distance <= self.radii["outer_bull"]:
            return "OB"
        elif self.radii["inner_treble"] < distance <= self.radii["outer_treble"]:
            return f"T{base_score}"
        elif self.radii["inner_double"] < distance <= self.radii["outer_double"]:
            return f"D{base_score}"
        elif distance <= self.radii["outer_double"]:
            return f"S{base_score}"
        else:
            return "Outside"

    def get_score_description(self, score):
        if score == "B":
            return "Bullseye! Worth 50 points."
        elif score == "OB":
            return "Outer Bull. Worth 25 points."
        elif score.startswith("T"):
            return f"Triple {score[1:]}. Worth {int(score[1:]) * 3} points."
        elif score.startswith("D"):
            return f"Double {score[1:]}. Worth {int(score[1:]) * 2} points."
        elif score.startswith("S"):
            return f"Single {score[1:]}. Worth {int(score[1:])} points."
        else:
            return "Outside the board. Worth 0 points."

    # -------------------------------
    # Camera Detection (using background subtraction and frame differencing)
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
            # Rotate frame 180° since camera is on the left
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

            # Reset detection data
            self.camera_data["dart_mm_y"] = None
            self.camera_data["dart_angle"] = None
            self.camera_data["tip_pixel"] = None

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tip_point = None
            if contours:
                # For left camera, choose the rightmost contour
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

            # Show debug windows (can be commented out in production)
            cv2.imshow("Camera ROI", roi)
            cv2.imshow("Combined Mask", combined_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # -------------------------------
    # Compute the camera vector based on detection
    # -------------------------------
    def compute_camera_vector(self):
        """
        Since a single camera only gives a ray, we define the camera-determined
        dart tip as the point 800mm (self.camera_vector_length) from the camera
        horizontally, with the board y-coordinate from calibration.
        """
        if self.camera_data.get("dart_mm_y") is None or self.camera_data.get("tip_pixel") is None:
            return None
        cam_x = self.camera_position[0]
        final_x = cam_x + self.camera_vector_length
        final_y = self.camera_data["dart_mm_y"]
        return (final_x, final_y)

    # -------------------------------
    # Apply lean offset to the camera vector (y-axis adjustment)
    # -------------------------------
    def apply_lean_offset(self, point):
        if point is None or self.current_cam_lean_angle is None:
            return point
        x, y = point
        deviation = 90 - self.current_cam_lean_angle  # positive means left lean
        offset = (deviation / 90.0) * self.side_lean_max_adjustment
        adjusted_y = y - offset  # For left camera: left lean => lower y
        return (x, adjusted_y)

    # -------------------------------
    # LIDAR helper functions (polar conversion, filtering, etc.)
    # -------------------------------
    def polar_to_cartesian(self, angle, distance, lidar_pos, rotation, mirror):
        if distance <= 0:
            return None, None
        angle_rad = np.radians(angle + rotation)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        if mirror:
            x = -x
        return x + lidar_pos[0], y + lidar_pos[1]

    def filter_points_by_radii(self, x, y):
        distance = np.sqrt(x**2 + y**2)
        for name, radius in self.radii.items():
            if distance <= radius:
                return True, name
        return False, None

    def apply_calibration_correction(self, x, y):
        if not self.calibration_points:
            return x, y
        total_weight = 0
        correction_x_weighted = 0
        correction_y_weighted = 0
        for ref_point, correction in self.calibration_points.items():
            ref_x, ref_y = ref_point
            dist = math.sqrt((x - ref_x)**2 + (y - ref_y)**2)
            if dist < 0.1:
                return x + correction[0], y + correction[1]
            weight = 1 / (dist * dist) if dist < 100 else 0
            correction_x_weighted += correction[0] * weight
            correction_y_weighted += correction[1] * weight
            total_weight += weight
        if total_weight > 0:
            return x + (correction_x_weighted / total_weight), y + (correction_y_weighted / total_weight)
        return x, y

    def get_wire_proximity_factor(self, x, y):
        dist = math.sqrt(x*x + y*y)
        ring_wire_threshold = 5.0
        min_ring_distance = float('inf')
        for radius in [self.radii["bullseye"], self.radii["outer_bull"],
                       self.radii["inner_treble"], self.radii["outer_treble"],
                       self.radii["inner_double"], self.radii["outer_double"]]:
            ring_distance = abs(dist - radius)
            min_ring_distance = min(min_ring_distance, ring_distance)
        angle_deg = math.degrees(math.atan2(y, x))
        if angle_deg < 0:
            angle_deg += 360
        min_segment_distance = float('inf')
        segment_boundary_degree = 9
        for i in range(20):
            boundary_angle = (segment_boundary_degree + i * 18) % 360
            angle_diff = min(abs(angle_deg - boundary_angle), 360 - abs(angle_deg - boundary_angle))
            linear_diff = angle_diff * math.pi / 180 * dist
            min_segment_distance = min(min_segment_distance, linear_diff)
        min_wire_distance = min(min_ring_distance, min_segment_distance)
        if min_wire_distance >= ring_wire_threshold:
            return 0.0
        else:
            return 1.0 - (min_wire_distance / ring_wire_threshold)

    def apply_segment_coefficients(self, x, y):
        wire_factor = self.get_wire_proximity_factor(x, y)
        angle_deg = math.degrees(math.atan2(y, x))
        if angle_deg < 0:
            angle_deg += 360
        segment = int(((angle_deg + 9) % 360) / 18) + 1
        radial_offset = self.segment_radial_offsets.get(segment, 0.0)
        if radial_offset != 0.0:
            magnitude = math.sqrt(x*x + y*y)
            if magnitude > 0:
                unit_x = x / magnitude
                unit_y = y / magnitude
                x += unit_x * radial_offset
                y += unit_y * radial_offset
        if wire_factor <= 0.0:
            return x, y
        dist = math.sqrt(x*x + y*y)
        if dist <= self.radii["outer_bull"]:
            return x, y
        angle_deg = math.degrees(math.atan2(y, x))
        if angle_deg < 0:
            angle_deg += 360
        segment = int(((angle_deg + 9) % 360) / 18) + 1
        seg_str = str(segment)
        if dist < self.radii["inner_treble"]:
            coeff_dict = self.small_segment_coeff
            ring_type = "small"
            possible_keys = [f"{seg_str}_1", f"{seg_str}_0", f"{seg_str}_5", f"{seg_str}_4"]
        elif dist < self.radii["outer_treble"]:
            coeff_dict = self.trebles_coeff
            ring_type = "trebles"
            possible_keys = [f"{seg_str}_1", f"{seg_str}_5", f"{seg_str}_0", f"{seg_str}_3", f"{seg_str}_2"]
        elif dist < self.radii["inner_double"]:
            coeff_dict = self.large_segment_coeff
            ring_type = "large"
            possible_keys = [f"{seg_str}_5", f"{seg_str}_4", f"{seg_str}_0", f"{seg_str}_1"]
        elif dist < self.radii["outer_double"]:
            coeff_dict = self.doubles_coeff
            ring_type = "doubles"
            possible_keys = [f"{seg_str}_1", f"{seg_str}_5", f"{seg_str}_0", f"{seg_str}_3"]
        else:
            return x, y
        scaling_factor = 1.0
        if segment in self.coefficient_scaling and ring_type in self.coefficient_scaling[segment]:
            scaling_factor = self.coefficient_scaling[segment][ring_type]
        for key in possible_keys:
            if key in coeff_dict:
                coeff = coeff_dict[key]
                correction_factor = wire_factor * scaling_factor
                return (x + (coeff["x_correction"] * correction_factor),
                        y + (coeff["y_correction"] * correction_factor))
        return x, y

    def detect_up_down_lean(self, lidar1_point, lidar2_point):
        if lidar1_point is not None and lidar2_point is not None:
            x1 = lidar1_point[0]
            x2 = lidar2_point[0]
            x_diff = x1 - x2
            lean_angle = (x_diff / self.MAX_X_DIFF_FOR_MAX_LEAN) * self.MAX_UP_DOWN_LEAN
            confidence = min(1.0, (len(self.lidar1_recent_points) + len(self.lidar2_recent_points)) / (2 * self.max_recent_points))
        elif (lidar1_point is not None or lidar2_point is not None) and self.camera_data.get("dart_angle") is not None:
            side_lean_angle = self.camera_data.get("dart_angle")
            side_lean_deviation = abs(90 - side_lean_angle)
            if lidar1_point is not None and lidar2_point is None:
                y_pos = lidar1_point[1]
                if side_lean_angle < 80 and y_pos > 0:
                    lean_angle = side_lean_deviation * 0.5
                elif side_lean_angle < 80 and y_pos <= 0:
                    lean_angle = side_lean_deviation * 0.3
                else:
                    lean_angle = 0
            elif lidar1_point is None and lidar2_point is not None:
                y_pos = lidar2_point[1]
                if side_lean_angle < 80 and y_pos < 0:
                    lean_angle = -side_lean_deviation * 0.5
                elif side_lean_angle < 80 and y_pos >= 0:
                    lean_angle = -side_lean_deviation * 0.3
                else:
                    lean_angle = 0
            else:
                lean_angle = 0
            confidence = min(0.8, side_lean_deviation / 90)
        else:
            lean_angle = 0
            confidence = 0
        lean_angle = max(-self.MAX_UP_DOWN_LEAN, min(self.MAX_UP_DOWN_LEAN, lean_angle))
        return lean_angle, confidence

    def project_lidar_point_with_3d_lean(self, lidar_point, lidar_height, side_lean_angle, up_down_lean_angle, camera_y):
        if lidar_point is None:
            return lidar_point
        if side_lean_angle is None:
            side_lean_angle = 90
        if up_down_lean_angle is None:
            up_down_lean_angle = 0
        original_x, original_y = lidar_point
        adjusted_y = original_y
        if camera_y is not None:
            side_lean_factor = side_lean_angle / 90.0
            inverse_side_lean = 1.0 - side_lean_factor
            y_displacement = original_y - camera_y
            MAX_SIDE_ADJUSTMENT = self.side_lean_max_adjustment
            side_adjustment = min(inverse_side_lean * abs(y_displacement), MAX_SIDE_ADJUSTMENT)
            side_adjustment *= -1 if y_displacement > 0 else 1
            adjusted_y = original_y + side_adjustment
        y_distance_from_center = abs(original_y)
        MAX_UP_DOWN_ADJUSTMENT = self.up_down_lean_max_adjustment
        up_down_adjustment = (up_down_lean_angle / 30.0) * (y_distance_from_center / 170.0) * MAX_UP_DOWN_ADJUSTMENT
        adjusted_x = original_x + up_down_adjustment
        return (adjusted_x, adjusted_y)

    def find_camera_board_intersection(self, camera_y):
        if camera_y is None:
            return None
        return (0, camera_y)

    def calculate_final_tip_position(self, camera_point, lidar1_point, lidar2_point):
        valid_points = []
        if camera_point is not None:
            valid_points.append(camera_point)
        if lidar1_point is not None:
            valid_points.append(lidar1_point)
        if lidar2_point is not None:
            valid_points.append(lidar2_point)
        if not valid_points:
            return None
        if len(valid_points) == 1:
            return valid_points[0]
        if camera_point is not None:
            if lidar1_point is not None and lidar2_point is not None:
                up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
                if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
                    if up_down_lean_angle > 0:
                        lidar_x = lidar1_point[0] * 0.7 + lidar2_point[0] * 0.3
                    else:
                        lidar_x = lidar1_point[0] * 0.3 + lidar2_point[0] * 0.7
                else:
                    lidar_x = (lidar1_point[0] + lidar2_point[0]) / 2
                final_x = lidar_x
                final_y = camera_point[1]
                final_tip_position = (final_x, final_y)
            elif lidar1_point is not None:
                final_tip_position = (lidar1_point[0], camera_point[1])
            elif lidar2_point is not None:
                final_tip_position = (lidar2_point[0], camera_point[1])
            else:
                final_tip_position = camera_point
        elif lidar1_point is not None and lidar2_point is not None:
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
            if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
                if up_down_lean_angle > 0:
                    weight1 = 0.7
                    weight2 = 0.3
                else:
                    weight1 = 0.3
                    weight2 = 0.7
                final_x = lidar1_point[0] * weight1 + lidar2_point[0] * weight2
                final_y = lidar1_point[1] * weight1 + lidar2_point[1] * weight2
            else:
                final_x = (lidar1_point[0] + lidar2_point[0]) / 2
                final_y = (lidar1_point[1] + lidar2_point[1]) / 2
            final_tip_position = (final_x, final_y)
        else:
            final_tip_position = valid_points[0]
        if final_tip_position is not None:
            x, y = final_tip_position
            x = x * self.x_scale_correction
            y = y * self.y_scale_correction
            final_tip_position = (x, y)
        return final_tip_position

    def update_lean_visualization(self, side_lean_angle, up_down_lean_angle, lean_confidence):
        arrow_length = 40
        arrow_x = -350
        arrow_y = 350
        if side_lean_angle is not None and up_down_lean_angle is not None:
            side_lean_rad = np.radians(90 - side_lean_angle)
            up_down_lean_rad = np.radians(up_down_lean_angle)
            dx = arrow_length * np.sin(side_lean_rad) * np.cos(up_down_lean_rad)
            dy = arrow_length * np.cos(side_lean_rad)
            if self.lean_arrow:
                self.lean_arrow.remove()
            self.lean_arrow = self.ax.arrow(
                arrow_x, arrow_y, dx, dy,
                head_width=8, head_length=10,
                fc='purple', ec='purple', alpha=0.7,
                length_includes_head=True
            )
            if self.arrow_text:
                self.arrow_text.remove()
            self.arrow_text = self.ax.text(
                arrow_x + dx + 10, arrow_y + dy,
                f"Conf: {lean_confidence:.2f}",
                fontsize=8, color='purple'
            )
        elif self.lean_arrow is None:
            self.lean_arrow = self.ax.arrow(
                arrow_x, arrow_y, 0, arrow_length,
                head_width=8, head_length=10,
                fc='gray', ec='gray', alpha=0.5,
                length_includes_head=True
            )

    # -------------------------------
    # Visualization Setup and Animation Update
    # -------------------------------
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_title("LIDAR and Camera Vector Visualization")
        self.ax.grid(True)
        self.update_dartboard_image()
        self.ax.plot(*self.lidar1_pos, "bo", label="LIDAR 1")
        self.ax.plot(*self.lidar2_pos, "go", label="LIDAR 2")
        self.ax.plot(*self.camera_position, "ro", label="Camera")
        for name, radius in self.radii.items():
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray', alpha=0.4)
            self.ax.add_patch(circle)
            self.ax.text(0, radius, name, color='gray', fontsize=8, ha='center', va='bottom')
        self.scatter1, = self.ax.plot([], [], "b.", label="LIDAR 1 Data", zorder=3)
        self.scatter2, = self.ax.plot([], [], "g.", label="LIDAR 2 Data", zorder=3)
        self.camera_vector, = self.ax.plot([], [], "r--", label="Camera Vector")
        self.lidar1_vector, = self.ax.plot([], [], "b--", label="LIDAR 1 Vector")
        self.lidar2_vector, = self.ax.plot([], [], "g--", label="LIDAR 2 Vector")
        self.camera_dart, = self.ax.plot([], [], "rx", markersize=8, label="Camera Intersection")
        self.lidar1_dart, = self.ax.plot([], [], "bx", markersize=8, label="LIDAR 1 Projected", zorder=3)
        self.lidar2_dart, = self.ax.plot([], [], "gx", markersize=8, label="LIDAR 2 Projected", zorder=3)
        self.detected_dart, = self.ax.plot([], [], "ro", markersize=4, label="Final Tip Position", zorder=10)
        self.lean_text = self.ax.text(-380, 380, "", fontsize=9)
        self.ax.legend(loc="upper right", fontsize=8)

    def update_dartboard_image(self):
        scaled_extent = [-170 * self.board_scale_factor, 170 * self.board_scale_factor,
                         -170 * self.board_scale_factor, 170 * self.board_scale_factor]
        self.ax.imshow(self.dartboard_image, extent=scaled_extent, zorder=0)

    def update_plot(self, frame):
        # Process camera data to compute the camera vector
        cam_vector = self.compute_camera_vector()
        adjusted_cam_vector = self.apply_lean_offset(cam_vector)
        if adjusted_cam_vector is not None and self.camera_data.get("tip_pixel") is not None:
            self.log_dart_data(
                adjusted_cam_vector,
                self.camera_data["tip_pixel"],
                self.current_cam_lean_angle if self.current_cam_lean_angle is not None else 0,
                self.current_cam_lean_direction
            )
            self.camera_vector.set_data(
                [self.camera_position[0], adjusted_cam_vector[0]],
                [self.camera_position[1], adjusted_cam_vector[1]]
            )
            self.camera_dart.set_data([adjusted_cam_vector[0]], [adjusted_cam_vector[1]])
        else:
            self.camera_vector.set_data([], [])
            self.camera_dart.set_data([], [])
        side_str = f"{self.current_cam_lean_angle:.1f}°" if self.current_cam_lean_angle is not None else "N/A"
        lean_text = f"Lean: {side_str} ({self.current_cam_lean_direction})"
        self.lean_text.set_text(lean_text)

        # Process LIDAR queues and recent points
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

        camera_y = self.camera_data["dart_mm_y"]
        side_lean_angle = self.camera_data["dart_angle"]
        up_down_lean_angle = 0
        lean_confidence = 0
        if len(self.lidar1_recent_points) > 0 and len(self.lidar2_recent_points) > 0:
            lidar1_point = self.lidar1_recent_points[-1]
            lidar2_point = self.lidar2_recent_points[-1]
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        self.update_lean_visualization(side_lean_angle, up_down_lean_angle, lean_confidence)
        camera_point = self.find_camera_board_intersection(camera_y)
        lidar1_projected = None
        lidar2_projected = None
        if len(self.lidar1_recent_points) > 0:
            lidar1_point = self.lidar1_recent_points[-1]
            lidar1_projected = self.project_lidar_point_with_3d_lean(
                lidar1_point, self.lidar1_height, side_lean_angle, up_down_lean_angle, camera_y
            )
        if len(self.lidar2_recent_points) > 0:
            lidar2_point = self.lidar2_recent_points[-1]
            lidar2_projected = self.project_lidar_point_with_3d_lean(
                lidar2_point, self.lidar2_height, side_lean_angle, up_down_lean_angle, camera_y
            )
        final_tip_position = self.calculate_final_tip_position(
            camera_point, lidar1_projected, lidar2_projected
        )
        if final_tip_position is not None:
            x, y = final_tip_position
            x, y = self.apply_segment_coefficients(x, y)
            x, y = self.apply_calibration_correction(x, y)
            final_tip_position = (x, y)
        self.log_dart_data(
            final_tip_position,
            self.camera_data["tip_pixel"],
            side_lean_angle,
            up_down_lean_angle
        )
        self.scatter1.set_data(lidar1_points_x, lidar1_points_y)
        self.scatter2.set_data(lidar2_points_x, lidar2_points_y)
        if camera_point is not None:
            self.camera_vector.set_data(
                [self.camera_position[0], camera_point[0]],
                [self.camera_position[1], camera_point[1]]
            )
            self.camera_dart.set_data([camera_point[0]], [camera_point[1]])
        else:
            self.camera_vector.set_data([], [])
            self.camera_dart.set_data([], [])
        if lidar1_projected is not None:
            self.lidar1_dart.set_data([lidar1_projected[0]], [lidar1_projected[1]])
        else:
            self.lidar1_dart.set_data([], [])
        if lidar2_projected is not None:
            self.lidar2_dart.set_data([lidar2_projected[0]], [lidar2_projected[1]])
        else:
            self.lidar2_dart.set_data([], [])
        if final_tip_position is not None:
            self.detected_dart.set_data([final_tip_position[0]], [final_tip_position[1]])
            score = self.xy_to_dartboard_score(final_tip_position[0], final_tip_position[1])
            if score != "Outside":
                description = self.get_score_description(score)
                if self.score_text:
                    self.score_text.set_text(description)
                else:
                    self.score_text = self.ax.text(-380, 360, description, fontsize=12, color='red')
        else:
            self.detected_dart.set_data([], [])
        if side_lean_angle is not None:
            side_lean_str = f"{side_lean_angle:.1f}°"
        else:
            side_lean_str = "N/A"
        if up_down_lean_angle is not None:
            up_down_lean_str = f"{up_down_lean_angle:.1f}°"
        else:
            up_down_lean_str = "N/A"
        lean_text = f"Side Lean: {side_lean_str}\nUp/Down: {up_down_lean_str}"
        self.lean_text.set_text(lean_text)
        artists = [self.scatter1, self.scatter2,
                   self.camera_vector, self.camera_dart,
                   self.lidar1_dart, self.lidar2_dart,
                   self.detected_dart, self.lean_text]
        if hasattr(self, 'score_text') and self.score_text:
            artists.append(self.score_text)
        if hasattr(self, 'lean_arrow') and self.lean_arrow:
            artists.append(self.lean_arrow)
        if hasattr(self, 'arrow_text') and self.arrow_text:
            artists.append(self.arrow_text)
        return artists

    # -------------------------------
    # LIDAR Thread (stubbed – adjust script paths as needed)
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
    # Main run loop
    # -------------------------------
    def run(self, lidar1_script, lidar2_script):
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
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            blit=True, interval=100,
            cache_frame_data=False
        )
        plt.show()

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

    # -------------------------------
    # Calibration Mode Methods
    # -------------------------------
    def calibration_mode(self):
        print("Calibration Mode")
        print("1. LIDAR Rotation Calibration")
        print("2. Coefficient Scaling Calibration")
        print("3. Segment Radial Offset Calibration")
        print("q. Quit")
        option = input("Select option: ")
        if option == "1":
            self._calibrate_lidar_rotation()
        elif option == "2":
            self._calibrate_coefficient_scaling()
        elif option == "3":
            self._calibrate_segment_radial_offsets()
        else:
            print("Exiting calibration mode.")

    def _calibrate_lidar_rotation(self):
        print("LIDAR Rotation Calibration Mode")
        print(f"Current LIDAR1 rotation: {self.lidar1_rotation}°")
        print(f"Current LIDAR2 rotation: {self.lidar2_rotation}°")
        while True:
            cmd = input("Enter L1+/L1-/L2+/L2- followed by degrees (e.g., L1+0.5) or 'q' to quit: ")
            if cmd.lower() == 'q':
                break
            try:
                if cmd.startswith("L1+"):
                    self.lidar1_rotation += float(cmd[3:])
                elif cmd.startswith("L1-"):
                    self.lidar1_rotation -= float(cmd[3:])
                elif cmd.startswith("L2+"):
                    self.lidar2_rotation += float(cmd[3:])
                elif cmd.startswith("L2-"):
                    self.lidar2_rotation -= float(cmd[3:])
                print(f"Updated LIDAR1 rotation: {self.lidar1_rotation}°")
                print(f"Updated LIDAR2 rotation: {self.lidar2_rotation}°")
            except:
                print("Invalid command format")

    def _calibrate_coefficient_scaling(self):
        print("Coefficient Scaling Calibration Mode")
        print("Adjust scaling factors for specific segments and ring types.")
        print("Format: [segment]:[ring_type]:[scale]")
        print("  - segment: 1-20 or 'all'")
        print("  - ring_type: 'doubles', 'trebles', 'small', 'large', or 'all'")
        print("Example: 20:doubles:1.5")
        print("Example: all:trebles:0.8")
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
                ring_types = []
                if ring_type.lower() == 'all':
                    ring_types = ['doubles', 'trebles', 'small', 'large']
                elif ring_type.lower() in ['doubles', 'trebles', 'small', 'large']:
                    ring_types = [ring_type.lower()]
                else:
                    print("Ring type must be 'doubles', 'trebles', 'small', 'large', or 'all'")
                    continue
                for segment in segments:
                    for rt in ring_types:
                        self.coefficient_scaling[segment][rt] = scale
                print(f"Updated scaling factors for {len(segments)} segment(s) and {len(ring_types)} ring type(s)")
            except ValueError:
                print("Scale must be a numeric value")

    def _calibrate_segment_radial_offsets(self):
        print("Segment Radial Offset Calibration Mode")
        print("Current segment radial offsets (mm):")
        for segment in range(1, 21):
            offset = self.segment_radial_offsets[segment]
            direction = "toward bull" if offset < 0 else "away from bull"
            if offset == 0:
                direction = "no offset"
            print(f"Segment {segment}: {abs(offset):.1f} mm {direction}")
        print("\nEnter commands in format: [segment]:[offset]")
        print("  - segment: 1-20 or 'all'")
        print("  - offset: value in mm (positive = away from bull, negative = toward bull)")
        while True:
            cmd = input("Enter offset command or 'q' to quit: ")
            if cmd.lower() == 'q':
                break
            try:
                parts = cmd.split(':')
                if len(parts) != 2:
                    print("Invalid format. Use segment:offset")
                    continue
                segment_str, offset_str = parts
                offset = float(offset_str)
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
                for segment in segments:
                    self.segment_radial_offsets[segment] = offset
                direction = "toward bull" if offset < 0 else "away from bull"
                if offset == 0:
                    direction = "no offset"
                print(f"Updated offset for {len(segments)} segment(s) to {abs(offset):.1f} mm {direction}")
            except ValueError:
                print("Offset must be a numeric value")

    # -------------------------------
    # Run and Signal Handling
    # -------------------------------
    def run(self, lidar1_script, lidar2_script):
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
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            blit=True, interval=100, cache_frame_data=False
        )
        plt.show()

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

    # -------------------------------
    # Methods to load/save calibration settings
    # -------------------------------
    def load_coefficient_scaling(self, filename="coefficient_scaling.json"):
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_scaling = json.load(f)
                self.coefficient_scaling = {int(k): v for k, v in loaded_scaling.items()}
                print(f"Coefficient scaling loaded from {filename}")
                return True
            else:
                print(f"Scaling file {filename} not found, using defaults")
                return False
        except Exception as e:
            print(f"Error loading coefficient scaling: {e}")
            return False

    def save_coefficient_scaling(self, filename="coefficient_scaling.json"):
        try:
            with open(filename, 'w') as f:
                json.dump(self.coefficient_scaling, f, indent=2)
            print(f"Coefficient scaling saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving coefficient scaling: {e}")
            return False

    def load_segment_radial_offsets(self, filename="segment_offsets.json"):
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_offsets = json.load(f)
                self.segment_radial_offsets = {int(k): v for k, v in loaded_offsets.items()}
                print(f"Segment radial offsets loaded from {filename}")
                return True
            else:
                print(f"Segment offsets file {filename} not found, using defaults")
                return False
        except Exception as e:
            print(f"Error loading segment radial offsets: {e}")
            return False

    def save_segment_radial_offsets(self, filename="segment_offsets.json"):
        try:
            with open(filename, 'w') as f:
                json.dump(self.segment_radial_offsets, f, indent=2)
            print(f"Segment radial offsets saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving segment radial offsets: {e}")
            return False

    # -------------------------------
    # Methods to set/get segment offsets
    # -------------------------------
    def set_segment_radial_offset(self, segment, offset_mm):
        if 1 <= segment <= 20:
            self.segment_radial_offsets[segment] = offset_mm
            print(f"Set segment {segment} radial offset to {offset_mm} mm")
        else:
            print(f"Invalid segment number: {segment}. Must be between 1-20.")

    def get_segment_radial_offset(self, segment):
        if 1 <= segment <= 20:
            return self.segment_radial_offsets[segment]
        else:
            print(f"Invalid segment number: {segment}. Must be between 1-20.")
            return 0.0

if __name__ == "__main__":
    lidar1_script = "./tri_test_lidar1"
    lidar2_script = "./tri_test_lidar2"
    visualizer = LidarCameraVisualizer()
    visualizer.load_coefficient_scaling()
    visualizer.load_segment_radial_offsets()
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            visualizer.calibration_mode()
            save = input("Save settings? (y/n): ")
            if save.lower() == 'y':
                visualizer.save_coefficient_scaling()
                visualizer.save_segment_radial_offsets()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python script.py                  - Run the program normally")
            print("  python script.py --calibrate      - Enter calibration mode")
            print("  python script.py --help           - Show this help message")
    else:
        visualizer.run(lidar1_script, lidar2_script)
