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
        self.camera_data = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}  # Tracking tip pixel too

        # ROI Settings for side camera
        self.camera_board_plane_y = 250  # Y-coordinate of board surface
        self.camera_roi_range = 30       # How much above and below to include
        self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
        self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
        self.camera_roi_left = 119       # Example left boundary
        self.camera_roi_right = 604      # Example right boundary

        # Calibration points for mapping pixel x-coordinate to board mm
        self.camera_calibration_points = []  # List of (pixel_x, mm_y) tuples
        self.pixel_to_mm_factor = -0.628  # Fallback linear slope (mm/pixel)
        self.pixel_offset = 192.8         # Fallback pixel offset

        # Detection persistence (to avoid flicker)
        self.last_valid_detection = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30

        # Background subtractor for camera detection
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=162, varThreshold=67, detectShadows=False
        )
        self.prev_gray = None  # For frame differencing
        self.diff_threshold = 25
        self.min_contour_area = 30

        # LIDAR queues and recent point storage
        self.lidar1_queue = Queue()
        self.lidar2_queue = Queue()
        self.lidar1_recent_points = []
        self.lidar2_recent_points = []
        self.max_recent_points = 20  # For smoothing

        # Projected LIDAR points after lean correction
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None
        self.camera_board_intersection = None

        # CSV logging initialization
        self.initialize_csv_logging()

        # Dartboard scoring configuration and image
        self.score_text = None
        self.board_scale_factor = 2.75
        self.dartboard_image = mpimg.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")

        # Radii for dartboard zones (mm)
        self.radii = {
            "bullseye": 6.35,
            "outer_bull": 15.9,
            "inner_treble": 99,
            "outer_treble": 107,
            "inner_double": 162,
            "outer_double": 170,
            "board_edge": 195,
        }

        # Segment radial offsets (default hard-coded)
        self.segment_radial_offsets = {}
        for segment in range(1, 21):
            self.segment_radial_offsets[segment] = -15  # mm

        # Coefficient dictionaries (hard coded defaults)
        self.large_segment_coeff = {
            "14_5": {"x_correction": -1.888, "y_correction": 12.790},
            "11_4": {"x_correction": -6.709, "y_correction": 14.045},
            "11_0": {"x_correction": -6.605, "y_correction": 13.916},
            "11_5": {"x_correction": -5.090, "y_correction": 11.821},
            "11_1": {"x_correction": -4.847, "y_correction": 11.024},
            "8_0": {"x_correction": -6.395, "y_correction": 13.293},
            "7_0": {"x_correction": -9.691, "y_correction": 12.377},
            "7_1": {"x_correction": -9.591, "y_correction": 11.792},
            "8_1": {"x_correction": -6.213, "y_correction": 11.840},
            "16_1": {"x_correction": -10.269, "y_correction": 11.723},
            "2_1": {"x_correction": -4.985, "y_correction": 10.641},
            "8_4": {"x_correction": -8.981, "y_correction": 11.649},
            "14_1": {"x_correction": -1.723, "y_correction": 10.927},
            "8_5": {"x_correction": -4.535, "y_correction": 11.181},
            "3_1": {"x_correction": -8.918, "y_correction": 10.807},
            "3_1": {"x_correction": 0.518, "y_correction": -131.807},
            "9_0": {"x_correction": -3.342, "y_correction": 7.611},
            "16_5": {"x_correction": -10.830, "y_correction": 10.470},
            "17_0": {"x_correction": -8.497, "y_correction": 8.977},
            "16_0": {"x_correction": -8.224, "y_correction": 8.937},
            "19_1": {"x_correction": -10.042, "y_correction": 8.801},
            "4_5": {"x_correction": -4.686, "y_correction": 8.744},
            "19_5": {"x_correction": -10.075, "y_correction": 8.604},
            "12_4": {"x_correction": -0.499, "y_correction": 8.492},
            "19_4": {"x_correction": -13.455, "y_correction": 8.033},
            "14_4": {"x_correction": -1.960, "y_correction": 7.888},
            "3_5": {"x_correction": -10.917, "y_correction": 7.779},
            "17_5": {"x_correction": -9.926, "y_correction": 5.224},
            "6_1": {"x_correction": -5.905, "y_correction": 7.375},
            "14_0": {"x_correction": -0.637, "y_correction": 7.264},
            "15_1": {"x_correction": -7.194, "y_correction": 6.566},
            "5_5": {"x_correction": -2.309, "y_correction": 5.927},
            "6_0": {"x_correction": -7.530, "y_correction": 5.270},
            "10_1": {"x_correction": -9.194, "y_correction": 6.558},
            "12_1": {"x_correction": -1.734, "y_correction": 4.059},
            "12_0": {"x_correction": -3.338, "y_correction": 6.216},
            "19_0": {"x_correction": -9.766, "y_correction": 6.125},
            "9_4": {"x_correction": -1.636, "y_correction": 4.641},
            "5_4": {"x_correction": -8.349, "y_correction": 5.671},
            "9_1": {"x_correction": -1.606, "y_correction": 5.520},
            "2_5": {"x_correction": -7.027, "y_correction": 4.896},
            "18_1": {"x_correction": -3.413, "y_correction": 4.881},
            "13_1": {"x_correction": -5.517, "y_correction": 3.166},
            "1_0": {"x_correction": -0.407, "y_correction": 4.015},
            "10_0": {"x_correction": -7.208, "y_correction": 3.562},
            "17_4": {"x_correction": -8.488, "y_correction": 4.264},
            "15_0": {"x_correction": -7.664, "y_correction": 3.148},
            "1_5": {"x_correction": -0.208, "y_correction": 3.515},
            "9_5": {"x_correction": -1.443, "y_correction": 4.024},
            "4_4": {"x_correction": 3.680, "y_correction": 3.977},
            "13_0": {"x_correction": -7.877, "y_correction": 3.825},
            "18_5": {"x_correction": -1.150, "y_correction": 2.951},
            "20_0": {"x_correction": -0.209, "y_correction": 3.703},
            "20_4": {"x_correction": 0.030, "y_correction": 3.679},
            "1_1": {"x_correction": 0.153, "y_correction": 3.588},
            "13_4": {"x_correction": -1.385, "y_correction": 3.558},
            "6_4": {"x_correction": -4.651, "y_correction": 3.224},
            "18_0": {"x_correction": 0.445, "y_correction": 3.093},
            "20_5": {"x_correction": 3.307, "y_correction": 1.799},
            "20_1": {"x_correction": 1.100, "y_correction": 2.753},
            "4_1": {"x_correction": -3.415, "y_correction": 3.065},
            "6_5": {"x_correction": -5.995, "y_correction": 2.865},
            "3_4": {"x_correction": -8.063, "y_correction": 2.598},
            "5_1": {"x_correction": -1.836, "y_correction": 2.499},
            "18_4": {"x_correction": -0.437, "y_correction": 2.494},
            "12_5": {"x_correction": -2.815, "y_correction": 1.152},
            "4_0": {"x_correction": -2.765, "y_correction": 1.995},
            "15_5": {"x_correction": -2.276, "y_correction": 1.083},
            "1_4": {"x_correction": -3.322, "y_correction": 1.726},
            "13_5": {"x_correction": -2.024, "y_correction": 0.790},
            "10_4": {"x_correction": -4.923, "y_correction": 0.783},
            "5_0": {"x_correction": -2.016, "y_correction": 0.725},
            "2_4": {"x_correction": -0.837, "y_correction": 0.118},
            "10_5": {"x_correction": -3.798, "y_correction": -0.596}
        }
        
        # Coefficients for the double ring area
        self.doubles_coeff = {
            "1_1": {"x_correction": 3.171, "y_correction": 0.025},
            "14_5": {"x_correction": 1.920, "y_correction": 6.191},
            "20_1": {"x_correction": 1.812, "y_correction": 1.691},
            "20_5": {"x_correction": 0.901, "y_correction": -0.772},
            "5_1": {"x_correction": 1.289, "y_correction": 2.309},
            "12_1": {"x_correction": -0.341, "y_correction": 2.428},
            "13_1": {"x_correction": -0.154, "y_correction": -2.869},
            "12_3": {"x_correction": 0.892, "y_correction": 1.668},
            "1_5": {"x_correction": 0.508, "y_correction": -2.633},
            "12_5": {"x_correction": 0.714, "y_correction": 0.641},
            "18_1": {"x_correction": 0.398, "y_correction": 0.107},
            "5_5": {"x_correction": 0.447, "y_correction": -0.214},
            "11_5": {"x_correction": 0.024, "y_correction": 10.893},
            "9_1": {"x_correction": -3.442, "y_correction": 4.916},
            "14_1": {"x_correction": -0.489, "y_correction": 11.128},
            "18_5": {"x_correction": -2.239, "y_correction": -0.348},
            "11_1": {"x_correction": -2.255, "y_correction": 11.938},
            "15_1": {"x_correction": -2.844, "y_correction": 5.225},
            "1_0": {"x_correction": -1.988, "y_correction": 0.472},
            "9_5": {"x_correction": -3.420, "y_correction": 3.773},
            "4_0": {"x_correction": -2.919, "y_correction": 9.090},
            "8_5": {"x_correction": -3.642, "y_correction": 14.419},
            "6_1": {"x_correction": -3.798, "y_correction": -2.819},
            "6_0": {"x_correction": -3.986, "y_correction": 0.326},
            "4_1": {"x_correction": -4.062, "y_correction": -0.001},
            "4_5": {"x_correction": -4.586, "y_correction": 2.522},
            "10_1": {"x_correction": -5.709, "y_correction": 0.317},
            "15_5": {"x_correction": -4.602, "y_correction": 6.307},
            "13_3": {"x_correction": -5.236, "y_correction": -2.793},
            "2_5": {"x_correction": -6.853, "y_correction": 4.826},
            "13_5": {"x_correction": -6.011, "y_correction": -0.742},
            "19_0": {"x_correction": -6.175, "y_correction": 7.132},
            "8_1": {"x_correction": -6.521, "y_correction": 12.622},
            "2_1": {"x_correction": -6.639, "y_correction": 4.853},
            "16_4": {"x_correction": -6.744, "y_correction": 10.677},
            "6_5": {"x_correction": -8.855, "y_correction": 1.088},
            "8_3": {"x_correction": -7.429, "y_correction": 12.300},
            "17_1": {"x_correction": -10.417, "y_correction": 2.648},
            "17_5": {"x_correction": -9.882, "y_correction": 9.256},
            "10_5": {"x_correction": -10.464, "y_correction": 2.446},
            "3_1": {"x_correction": -12.581, "y_correction": 8.704}
        }
        
        # Coefficients for the treble ring area
        self.trebles_coeff = {
            "1_1": {"x_correction": 3.916, "y_correction": 7.238},
            "1_5": {"x_correction": 2.392, "y_correction": 0.678},
            "20_5": {"x_correction": 0.486, "y_correction": 4.293},
            "12_5": {"x_correction": -3.547, "y_correction": 6.943},
            "9_5": {"x_correction": -2.731, "y_correction": 6.631},
            "5_3": {"x_correction": 0.329, "y_correction": 6.408},
            "18_5": {"x_correction": -0.707, "y_correction": 0.318},
            "6_5": {"x_correction": -3.776, "y_correction": 0.478},
            "5_4": {"x_correction": -0.643, "y_correction": 5.413},
            "20_4": {"x_correction": -1.589, "y_correction": 4.333},
            "4_5": {"x_correction": -2.487, "y_correction": -0.736},
            "1_4": {"x_correction": -2.523, "y_correction": 5.148},
            "1_3": {"x_correction": -1.564, "y_correction": 4.005},
            "11_4": {"x_correction": -1.692, "y_correction": 12.002},
            "4_4": {"x_correction": -2.392, "y_correction": 5.377},
            "18_2": {"x_correction": -2.261, "y_correction": 2.238},
            "15_4": {"x_correction": -4.329, "y_correction": 6.308},
            "14_5": {"x_correction": -4.835, "y_correction": 9.586},
            "13_5": {"x_correction": -3.615, "y_correction": 2.233},
            "15_3": {"x_correction": -2.650, "y_correction": -0.747},
            "14_3": {"x_correction": -2.933, "y_correction": 10.375},
            "5_5": {"x_correction": -3.065, "y_correction": 5.476},
            "13_2": {"x_correction": -3.526, "y_correction": 5.453},
            "2_4": {"x_correction": -3.536, "y_correction": 2.872},
            "18_4": {"x_correction": -3.965, "y_correction": 2.129},
            "10_2": {"x_correction": -4.432, "y_correction": -3.304},
            "11_5": {"x_correction": -7.210, "y_correction": 9.992},
            "16_4": {"x_correction": -4.816, "y_correction": 6.071},
            "10_4": {"x_correction": -6.640, "y_correction": 4.376},
            "8_5": {"x_correction": -5.485, "y_correction": 11.023},
            "6_2": {"x_correction": -5.893, "y_correction": 5.692},
            "13_4": {"x_correction": -6.091, "y_correction": 5.513},
            "6_4": {"x_correction": -6.493, "y_correction": 2.749},
            "8_3": {"x_correction": -6.553, "y_correction": 10.721},
            "15_5": {"x_correction": -7.578, "y_correction": 5.281},
            "17_4": {"x_correction": -7.701, "y_correction": 4.141},
            "9_4": {"x_correction": -7.743, "y_correction": 5.096},
            "10_3": {"x_correction": -7.774, "y_correction": 0.795},
            "2_5": {"x_correction": -9.829, "y_correction": 7.675},
            "2_3": {"x_correction": -9.342, "y_correction": 4.183},
            "17_3": {"x_correction": -9.878, "y_correction": 8.301},
            "7_5": {"x_correction": -10.593, "y_correction": 11.340},
            "3_2": {"x_correction": -10.107, "y_correction": 10.510},
            "19_2": {"x_correction": -10.599, "y_correction": 5.780},
            "10_5": {"x_correction": -10.654, "y_correction": 2.223},
            "7_3": {"x_correction": -10.696, "y_correction": 11.291},
            "16_3": {"x_correction": -11.650, "y_correction": 8.589},
            "3_3": {"x_correction": -12.353, "y_correction": 8.187}
        }

        # Calibration correction using a weighted interpolation of predefined points
        self.calibration_points = {
            (0, 0): (-1.0, 0),
            (23, 167): (0.9, 2.7),
            (-23, -167): (0.4, 0.6),
            (167, -24): (2.9, 2.6),
            (-167, 24): (-3.3, -2.7),
            (75, 75): (2.3, 2.1),
            (-75, -75): (2.2, -3.2)
        }
        
        self.x_scale_correction = 1.02
        self.y_scale_correction = 1.04
        
        # 3D lean detection variables
        self.current_up_down_lean_angle = 0.0
        self.up_down_lean_confidence = 0.0
        self.lean_history = []
        self.max_lean_history = 60
        self.lean_arrow = None
        self.arrow_text = None
        
        self.MAX_SIDE_LEAN = 35.0
        self.MAX_UP_DOWN_LEAN = 30.0
        self.MAX_X_DIFF_FOR_MAX_LEAN = 4.0
        
        self.setup_plot()
        signal.signal(signal.SIGINT, self.signal_handler)

    def xy_to_dartboard_score(self, x, y):
        scores = [13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6]
        distance = np.sqrt(x * x + y * y)
        angle = np.degrees(np.arctan2(y, x))
        angle = (angle - 9 + 360) % 360
        segment_idx = int(angle / 18)
        if segment_idx >= len(scores):
            return "Outside"
        base_score = scores[segment_idx]
        if distance <= self.radii['bullseye']:
            return "B"
        elif distance <= self.radii['outer_bull']:
            return "OB"
        elif self.radii['inner_treble'] < distance <= self.radii['outer_treble']:
            return f"T{base_score}"
        elif self.radii['inner_double'] < distance <= self.radii['outer_double']:
            return f"D{base_score}"
        elif distance <= self.radii['outer_double']:
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

    def initialize_csv_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"dart_data_{timestamp}.csv"
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Dart_X_mm', 'Dart_Y_mm', 
                                 'Tip_Pixel_X', 'Tip_Pixel_Y', 
                                 'Side_Lean_Angle', 'Up_Down_Lean_Angle',
                                 'Score'])
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

    def add_calibration_point(self, pixel_x, mm_y):
        self.camera_calibration_points.append((pixel_x, mm_y))
        self.camera_calibration_points.sort(key=lambda p: p[0])
        print(f"Added calibration point: pixel_x={pixel_x}, mm_y={mm_y}")
        print(f"Current calibration points: {self.camera_calibration_points}")

    def pixel_to_mm(self, pixel_x):
        if len(self.camera_calibration_points) >= 2:
            for i in range(len(self.camera_calibration_points) - 1):
                p1_pixel, p1_mm = self.camera_calibration_points[i]
                p2_pixel, p2_mm = self.camera_calibration_points[i + 1]
                if p1_pixel <= pixel_x <= p2_pixel:
                    return p1_mm + (pixel_x - p1_pixel) * (p2_mm - p1_mm) / (p2_pixel - p1_pixel)
            if pixel_x < self.camera_calibration_points[0][0]:
                return self.camera_calibration_points[0][1]
            else:
                return self.camera_calibration_points[-1][1]
        else:
            return self.pixel_to_mm_factor * pixel_x + self.pixel_offset

    def clear_calibration_points(self):
        self.camera_calibration_points = []
        print("Calibration points cleared.")

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

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

    def polar_to_cartesian(self, angle, distance, lidar_pos, rotation, mirror):
        if distance <= 0:
            return None, None
        angle_rad = np.radians(angle + rotation)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        if mirror:
            x = -x
        return x + lidar_pos[0], y + lidar_pos[1]

    def measure_tip_angle(self, mask, tip_point):
        if tip_point is None:
            return None
        tip_x, tip_y = tip_point
        search_depth = 25
        search_width = 40
        min_points = 6
        min_x = max(0, tip_x)
        max_x = min(mask.shape[1] - 1, tip_x + search_depth)
        min_y = max(0, tip_y - search_width)
        max_y = min(mask.shape[0] - 1, tip_y + search_width)
        points_right = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if mask[y, x] > 0:
                    points_right.append((x, y))
        if len(points_right) < min_points:
            return None
        best_angle = None
        best_inliers = 0
        for _ in range(10):
            if len(points_right) < 2:
                continue
            indices = np.random.choice(len(points_right), 2, replace=False)
            p1 = points_right[indices[0]]
            p2 = points_right[indices[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 5:
                continue
            if dx == 0:
                angle = 90
            else:
                slope = dy / dx
                angle_from_horizontal = math.degrees(math.atan(slope))
                angle = 90 - angle_from_horizontal
            inliers = []
            for point in points_right:
                if dx == 0:
                    dist_to_line = abs(point[0] - p1[0])
                else:
                    a = -slope
                    b = 1
                    c = slope * p1[0] - p1[1]
                    dist_to_line = abs(a*point[0] + b*point[1] + c) / math.sqrt(a*a + b*b)
                if dist_to_line < 2:
                    inliers.append(point)
            if len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_angle = angle
        if best_angle is None:
            points = np.array(points_right)
            if len(points) < 2:
                return None
            x_vals = points[:, 0]
            y_vals = points[:, 1]
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
            denominator = np.sum((x_vals - x_mean)**2)
            if denominator == 0:
                best_angle = 90
            else:
                slope = numerator / denominator
                angle_from_horizontal = math.degrees(math.atan(slope))
                best_angle = 90 - angle_from_horizontal
        return best_angle

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
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
            self.camera_data["dart_mm_y"] = None
            self.camera_data["dart_angle"] = None
            self.camera_data["tip_pixel"] = None
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                tip_contour = None
                tip_point = None
                for contour in contours:
                    if cv2.contourArea(contour) > self.min_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        dart_pixel_x = x + w // 2
                        roi_center_y = self.camera_board_plane_y - self.camera_roi_top
                        if tip_contour is None:
                            tip_contour = contour
                            tip_point = (dart_pixel_x, roi_center_y)
                if tip_contour is not None and tip_point is not None:
                    dart_angle = self.measure_tip_angle(combined_mask, tip_point)
                    global_pixel_x = tip_point[0] + self.camera_roi_left
                    global_pixel_y = tip_point[1] + self.camera_roi_top
                    dart_mm_y = self.pixel_to_mm(global_pixel_x)
                    self.camera_data["dart_mm_y"] = dart_mm_y
                    self.camera_data["dart_angle"] = dart_angle
                    self.camera_data["tip_pixel"] = (global_pixel_x, global_pixel_y)
                    self.last_valid_detection = self.camera_data.copy()
                    self.detection_persistence_counter = self.detection_persistence_frames
            elif self.detection_persistence_counter > 0:
                self.detection_persistence_counter -= 1
                if self.detection_persistence_counter > 0:
                    self.camera_data = self.last_valid_detection.copy()
        cap.release()

    def apply_calibration_correction(self, x, y):
        if not self.calibration_points:
            return x, y
        total_weight = 0.0
        weighted_corr_x = 0.0
        weighted_corr_y = 0.0
        for ref_point, correction in self.calibration_points.items():
            ref_x, ref_y = ref_point
            dist = math.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2)
            if dist < 0.1:
                return x + correction[0], y + correction[1]
            weight = 1.0 / (dist * dist)
            weighted_corr_x += correction[0] * weight
            weighted_corr_y += correction[1] * weight
            total_weight += weight
        if total_weight > 0:
            corr_x = weighted_corr_x / total_weight
            corr_y = weighted_corr_y / total_weight
            return x + corr_x, y + corr_y
        return x, y

    def get_wire_proximity_factor(self, x, y):
        dist = math.sqrt(x*x + y*y)
        ring_wire_threshold = 5.0  # mm
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
        ring_type = ""
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
                return (x + coeff["x_correction"] * correction_factor, 
                        y + coeff["y_correction"] * correction_factor)
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
        MAX_DISCREPANCY = 15.0  # mm
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
                arrow_x + dx + 10, arrow_y + dy, f"Conf: {lean_confidence:.2f}", fontsize=8, color='purple'
            )
        elif self.lean_arrow is None:
            self.lean_arrow = self.ax.arrow(
                arrow_x, arrow_y, 0, arrow_length, 
                head_width=8, head_length=10, 
                fc='gray', ec='gray', alpha=0.5,
                length_includes_head=True
            )

    def update_plot(self, frame):
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
            lidar1_projected = self.project_lidar_point_with_3d_lean(lidar1_point, self.lidar1_height, side_lean_angle, up_down_lean_angle, camera_y)
        if len(self.lidar2_recent_points) > 0:
            lidar2_point = self.lidar2_recent_points[-1]
            lidar2_projected = self.project_lidar_point_with_3d_lean(lidar2_point, self.lidar2_height, side_lean_angle, up_down_lean_angle, camera_y)
        final_tip_position = self.calculate_final_tip_position(camera_point, lidar1_projected, lidar2_projected)
        if final_tip_position is not None:
            x, y = final_tip_position
            x, y = self.apply_segment_coefficients(x, y)
            x, y = self.apply_calibration_correction(x, y)
            final_tip_position = (x, y)
        self.log_dart_data(final_tip_position, self.camera_data["tip_pixel"], side_lean_angle, up_down_lean_angle)
        self.scatter1.set_data(lidar1_points_x, lidar1_points_y)
        self.scatter2.set_data(lidar2_points_x, lidar2_points_y)
        if camera_point is not None:
            self.camera_vector.set_data([self.camera_position[0], camera_point[0]], [self.camera_position[1], camera_point[1]])
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
        artists = [self.scatter1, self.scatter2, self.camera_vector, self.camera_dart,
                   self.lidar1_dart, self.lidar2_dart, self.detected_dart, self.lean_text]
        if hasattr(self, 'score_text') and self.score_text:
            artists.append(self.score_text)
        if hasattr(self, 'lean_arrow') and self.lean_arrow:
            artists.append(self.lean_arrow)
        if hasattr(self, 'arrow_text') and self.arrow_text:
            artists.append(self.arrow_text)
        return artists

    def run(self, lidar1_script, lidar2_script):
        lidar1_thread = threading.Thread(target=self.start_lidar, args=(lidar1_script, self.lidar1_queue, 1), daemon=True)
        lidar2_thread = threading.Thread(target=self.start_lidar, args=(lidar2_script, self.lidar2_queue, 2), daemon=True)
        camera_thread = threading.Thread(target=self.camera_detection, daemon=True)
        lidar1_thread.start()
        time.sleep(1)
        lidar2_thread.start()
        time.sleep(1)
        camera_thread.start()
        self.ani = FuncAnimation(self.fig, self.update_plot, blit=True, interval=100, cache_frame_data=False)
        plt.show()

    # NEW: Method to set segment radial offset
    def set_segment_radial_offset(self, segment, offset_mm):
        if 1 <= segment <= 20:
            self.segment_radial_offsets[segment] = offset_mm
            print(f"Set segment {segment} radial offset to {offset_mm} mm")
        else:
            print(f"Invalid segment number: {segment}. Must be between 1-20.")

    # NEW: Method to get segment radial offset
    def get_segment_radial_offset(self, segment):
        if 1 <= segment <= 20:
            return self.segment_radial_offsets[segment]
        else:
            print(f"Invalid segment number: {segment}. Must be between 1-20.")
            return 0.0

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

    # Calibration mode methods
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
        print("  - scale: scaling factor (e.g. 0.5, 1.0, 1.5)")
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
                if len(segments) <= 3:
                    for segment in segments:
                        print(f"Segment {segment}: " + ", ".join([f"{rt}={self.coefficient_scaling[segment][rt]}" for rt in ring_types]))
                else:
                    print(f"Set {', '.join(ring_types)} scaling factor to {scale} for segments {segments[0]}-{segments[-1]}")
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
        print("Example: 3:11")
        print("Example: 6:-5")
        print("Example: all:0")
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
