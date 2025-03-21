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

        # Camera configuration
        self.camera_position = (0, 350)     # Camera is above the board
        self.camera_vector_length = 1600     # Vector length in mm
        self.camera_data = {"dart_mm_x": None, "dart_angle": None}

        # ROI Settings and Pixel-to-mm Mapping
        self.roi_top = 148  # Top of the ROI
        self.roi_bottom = 185  # Bottom of the ROI
        self.pixel_to_mm_x = (180 - (-180)) / (556 - 126)  # Calibrated conversion
        self.camera_x_offset = 9.8  # Small offset to account for systematic error

        # Detection persistence to maintain visibility
        self.last_valid_detection = {"dart_mm_x": None, "dart_angle": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30

        # Camera background subtractor with sensitive parameters
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )

        # LIDAR queues
        self.lidar1_queue = Queue()
        self.lidar2_queue = Queue()
        
        # Storage for most recent LIDAR data points
        self.lidar1_recent_points = []
        self.lidar2_recent_points = []
        self.max_recent_points = 20  # Keep last 5 points for smoothing

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

        # --- Coefficient dictionaries ---
        # Coefficients for the outer single area (large segments)
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
            "10_5": {"x_correction": -3.798, "y_correction": -0.596},
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
            "3_1": {"x_correction": -12.581, "y_correction": 8.704},
        }
        
        # Coefficients for the triple ring area (trebles)
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
            "3_3": {"x_correction": -12.353, "y_correction": 8.187},
        }
        
        # Coefficients for the inner single area (small segments)
        self.small_segment_coeff = {
            "8_5": {"x_correction": -7.021, "y_correction": 9.646},
            "5_1": {"x_correction": -2.830, "y_correction": 9.521},
            "11_5": {"x_correction": -6.008, "y_correction": 10.699},
            "2_5": {"x_correction": -6.268, "y_correction": 7.615},
            "20_1": {"x_correction": 0.963, "y_correction": 10.550},
            "14_4": {"x_correction": -5.388, "y_correction": 8.898},
            "9_4": {"x_correction": -5.635, "y_correction": 8.149},
            "12_0": {"x_correction": -6.959, "y_correction": 10.260},
            "16_5": {"x_correction": -11.642, "y_correction": 10.048},
            "14_5": {"x_correction": -4.386, "y_correction": 10.033},
            "3_4": {"x_correction": -7.878, "y_correction": 9.123},
            "19_4": {"x_correction": -8.770, "y_correction": 9.305},
            "17_5": {"x_correction": -8.988, "y_correction": 9.779},
            "5_5": {"x_correction": 1.819, "y_correction": 9.487},
            "9_0": {"x_correction": -4.911, "y_correction": 7.613},
            "4_1": {"x_correction": -1.519, "y_correction": 8.133},
            "7_4": {"x_correction": -11.424, "y_correction": 9.196},
            "16_4": {"x_correction": -10.080, "y_correction": 8.196},
            "12_5": {"x_correction": -1.101, "y_correction": 8.018},
            "14_0": {"x_correction": -6.371, "y_correction": 8.618},
            "20_4": {"x_correction": -2.488, "y_correction": 7.384},
            "18_0": {"x_correction": 0.999, "y_correction": 7.666},
            "12_1": {"x_correction": -2.311, "y_correction": 7.972},
            "11_1": {"x_correction": -5.123, "y_correction": 8.361},
            "7_5": {"x_correction": -10.936, "y_correction": 8.215},
            "1_0": {"x_correction": -1.665, "y_correction": 8.301},
            "1_1": {"x_correction": 1.706, "y_correction": 8.171},
            "17_4": {"x_correction": -10.220, "y_correction": 7.089},
            "19_5": {"x_correction": -9.638, "y_correction": 8.098},
            "8_1": {"x_correction": -6.475, "y_correction": 7.005},
            "18_5": {"x_correction": -1.040, "y_correction": 5.177},
            "18_1": {"x_correction": -0.922, "y_correction": 6.026},
            "6_4": {"x_correction": 0.032, "y_correction": 7.757},
            "10_0": {"x_correction": -0.047, "y_correction": 4.356},
            "1_5": {"x_correction": -1.026, "y_correction": 7.089},
            "7_0": {"x_correction": -2.914, "y_correction": 6.025},
            "19_0": {"x_correction": -4.016, "y_correction": 6.535},
            "3_1": {"x_correction": -3.210, "y_correction": 7.153},
            "11_4": {"x_correction": -5.380, "y_correction": 7.119},
            "6_0": {"x_correction": -0.378, "y_correction": 4.360},
            "15_1": {"x_correction": -2.137, "y_correction": 5.343},
            "2_1": {"x_correction": -3.566, "y_correction": 6.891},
            "13_5": {"x_correction": -1.611, "y_correction": 5.200},
            "8_0": {"x_correction": -5.113, "y_correction": 6.868},
            "10_4": {"x_correction": -5.279, "y_correction": 4.516},
            "18_4": {"x_correction": -1.608, "y_correction": 6.575},
            "16_0": {"x_correction": -2.293, "y_correction": 6.462},
            "4_0": {"x_correction": -1.653, "y_correction": 6.338},
            "5_4": {"x_correction": -2.519, "y_correction": 4.993},
            "12_4": {"x_correction": -3.529, "y_correction": 6.306},
            "13_4": {"x_correction": -2.516, "y_correction": 5.167},
            "4_5": {"x_correction": -2.475, "y_correction": 6.135},
        }

        # Calibration factors for lean correction
        self.side_lean_max_adjustment = 6.0  # mm, maximum adjustment for side lean
        self.forward_lean_max_adjustment = 4.0  # mm, maximum adjustment for forward lean
        
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
        self.current_forward_lean_angle = 0.0
        self.forward_lean_confidence = 0.0
        self.lean_history = []  # Store recent lean readings for smoothing
        self.max_lean_history = 60  # Keep track of last 10 lean readings
        self.lean_arrow = None  # For visualization
        self.arrow_text = None  # For visualization text
        
        # Maximum expected lean angles
        self.MAX_SIDE_LEAN = 35.0  # Maximum expected side-to-side lean in degrees
        self.MAX_FORWARD_LEAN = 30.0  # Maximum expected forward/backward lean in degrees
        
        # Maximum expected Y-difference for maximum lean (calibration parameter)
        self.MAX_Y_DIFF_FOR_MAX_LEAN = 4.0  # mm
        
        # Setup visualization
        self.setup_plot()

        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)

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

    def calculate_dart_angle(self, contour):
        """Calculate the angle of the dart tip relative to vertical.
        Returns angle in degrees where:
        - 90 degrees = perfectly upright (perpendicular to board)
        - 0 degrees = flat against the board (parallel)
        """
        if len(contour) < 5:
            return None
            
        # Fit an ellipse to the contour
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Convert to 0-90 degrees relative to vertical (90° = vertical)
            if angle > 90:
                angle = 180 - angle
                
            return angle
        except:
            return None

    def camera_detection(self):
        """Detect dart tip using the camera."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the frame 180 degrees since camera is upside down
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            roi = frame[self.roi_top:self.roi_bottom, :]

            # Background subtraction and thresholding with enhanced parameters
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            fg_mask = self.camera_bg_subtractor.apply(gray)
            
            # More sensitive threshold
            fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)[1]
            
            # Morphological operations to enhance the dart
            kernel = np.ones((3,3), np.uint8)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

            # Reset current detection
            self.camera_data["dart_mm_x"] = None
            self.camera_data["dart_angle"] = None

            # Detect contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the dart tip (highest point since image is flipped)
                tip_contour = None
                lowest_point = (-1, -1)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 50:  # Reduced threshold to catch smaller darts
                        for point in contour:
                            x, y = point[0]
                            if tip_contour is None or y < lowest_point[1]:
                                lowest_point = (x, y)
                                tip_contour = contour
                
                if tip_contour is not None:
                    # Get dart angle
                    dart_angle = self.calculate_dart_angle(tip_contour)
                    
                    # Map pixels to mm coordinates with corrected mapping
                    tip_pixel_x = lowest_point[0]
                    dart_mm_x = 180 - (tip_pixel_x - 126) * self.pixel_to_mm_x + self.camera_x_offset
                    
                    # Save data
                    self.camera_data["dart_mm_x"] = dart_mm_x
                    self.camera_data["dart_angle"] = dart_angle
                    
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

        # Get the scaling factor specific to this segment and ring
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

    def detect_forward_backward_lean(self, lidar1_point, lidar2_point):
        """
        Enhanced forward/backward lean detection that works with one or two LIDARs.
        
        Args:
            lidar1_point: (x, y) position from LIDAR 1
            lidar2_point: (x, y) position from LIDAR 2
            
        Returns:
            lean_angle: Estimated lean angle in degrees (0° = vertical, positive = toward LIDAR 1, negative = toward LIDAR 2)
            confidence: Confidence level in the lean detection (0-1)
        """
        # Case 1: Both LIDARs active - use standard Y-difference calculation
        if lidar1_point is not None and lidar2_point is not None:
            # Extract y-coordinates from both LIDARs
            y1 = lidar1_point[1]
            y2 = lidar2_point[1]
            
            # Calculate y-difference as indicator of forward/backward lean
            y_diff = y1 - y2
            
            # Convert y-difference to an angle
            lean_angle = (y_diff / self.MAX_Y_DIFF_FOR_MAX_LEAN) * self.MAX_FORWARD_LEAN
            
            # Calculate confidence based on how many points we have
            confidence = min(1.0, (len(self.lidar1_recent_points) + len(self.lidar2_recent_points)) / 
                             (2 * self.max_recent_points))
            
        # Case 2: Single LIDAR with significant side lean - infer forward lean
        elif (lidar1_point is not None or lidar2_point is not None) and self.camera_data.get("dart_angle") is not None:
            side_lean_angle = self.camera_data.get("dart_angle")
            
            # Calculate how far from vertical the side lean is
            side_lean_deviation = abs(90 - side_lean_angle)
            
            # Only LIDAR 1 (left side) active
            if lidar1_point is not None and lidar2_point is None:
                # Get quadrant information
                x_pos = lidar1_point[0]
                
                # If leaning left and on left side: likely toward LIDAR 1
                if side_lean_angle < 80 and x_pos < 0:
                    # More side lean = more forward lean
                    lean_angle = side_lean_deviation * 0.5
                # If leaning left and on right side: likely away from LIDAR 2
                elif side_lean_angle < 80 and x_pos >= 0:
                    lean_angle = side_lean_deviation * 0.3
                # Near vertical or anomalous position
                else:
                    lean_angle = 0
            
            # Only LIDAR 2 (right side) active
            elif lidar1_point is None and lidar2_point is not None:
                # Get position information
                x_pos = lidar2_point[0]
                
                # If leaning left and on right side: likely toward LIDAR 2 (negative lean)
                if side_lean_angle < 80 and x_pos > 0:
                    lean_angle = -side_lean_deviation * 0.5
                # If leaning left and on left side: likely away from LIDAR 1 (negative lean)
                elif side_lean_angle < 80 and x_pos <= 0:
                    lean_angle = -side_lean_deviation * 0.3
                # Near vertical or anomalous position
                else:
                    lean_angle = 0
            
            # No LIDARs (shouldn't happen in this function but handle it)
            else:
                lean_angle = 0
                
            # Special case for diagonal leans
            # If camera shows dart leaning severely left (<30°) and right LIDAR sees it:
            # Dart must be leaning toward camera (forward)
            if side_lean_angle < 30 and lidar2_point is not None:
                lean_angle = abs(lean_angle)  # Force positive (toward camera)
                
            # If camera shows dart leaning severely right (>150°) and left LIDAR sees it:
            # Dart must be leaning toward camera (forward)
            elif side_lean_angle > 150 and lidar1_point is not None:
                lean_angle = abs(lean_angle)  # Force positive (toward camera)
                
            # Use a scaled confidence based on side lean deviation
            confidence = min(0.8, side_lean_deviation / 90)
            
        # Case 3: No valid data
        else:
            lean_angle = 0
            confidence = 0
        
        # Clamp to reasonable range
        lean_angle = max(-self.MAX_FORWARD_LEAN, min(self.MAX_FORWARD_LEAN, lean_angle))
        
        return lean_angle, confidence
    def project_lidar_point_with_3d_lean(self, lidar_point, lidar_height, side_lean_angle, forward_lean_angle, camera_x):
        """
        Project a LIDAR detection point to account for both side-to-side and forward/backward lean.
        
        Args:
            lidar_point: (x, y) position of LIDAR detection
            lidar_height: height of the LIDAR beam above board in mm
            side_lean_angle: angle of dart from vertical in degrees (90° = vertical, from camera)
            forward_lean_angle: angle of forward/backward lean in degrees (0° = vertical)
            camera_x: X-coordinate from camera detection
            
        Returns:
            (x, y) position of the adjusted point
        """
        if lidar_point is None:
            return lidar_point
            
        # Handle missing lean angles
        if side_lean_angle is None:
            side_lean_angle = 90  # Default to vertical
        if forward_lean_angle is None:
            forward_lean_angle = 0  # Default to vertical
        
        # Extract original coordinates
        original_x, original_y = lidar_point
        
        # Apply side-to-side lean adjustment if we have camera data
        adjusted_x = original_x
        if camera_x is not None:
            # Calculate side-to-side lean adjustment (similar to your existing code)
            # Convert to 0-1 scale where 0 is horizontal (0°) and 1 is vertical (90°)
            side_lean_factor = side_lean_angle / 90.0
            inverse_side_lean = 1.0 - side_lean_factor
            
            # Calculate X displacement (how far LIDAR point is from camera line)
            x_displacement = original_x - camera_x
            
            # Apply side-to-side adjustment proportional to lean angle, with constraints
            MAX_SIDE_ADJUSTMENT = self.side_lean_max_adjustment  # mm
            side_adjustment = min(inverse_side_lean * abs(x_displacement), MAX_SIDE_ADJUSTMENT)
            side_adjustment *= -1 if x_displacement > 0 else 1
            
            # Apply side-to-side adjustment
            adjusted_x = original_x + side_adjustment
        
        # Calculate forward/backward adjustment
        # This is proportional to the forward/backward lean angle and the distance from board center
        # The farther from center, the more adjustment needed for the same lean angle
        
        # Distance from board center in X direction
        x_distance_from_center = abs(original_x)
        
        # Calculate forward/backward adjustment 
        # Positive forward_lean_angle means leaning toward LIDAR 1, adjustment should be in Y direction
        MAX_FORWARD_ADJUSTMENT = self.forward_lean_max_adjustment  # mm
        
        # Calculate adjustment proportional to lean angle and distance from center
        # More lean and further from center = bigger adjustment
        forward_adjustment = (forward_lean_angle / 30.0) * (x_distance_from_center / 170.0) * MAX_FORWARD_ADJUSTMENT
        
        # Apply forward/backward adjustment to Y coordinate
        adjusted_y = original_y + forward_adjustment
        
        return (adjusted_x, adjusted_y)

    def find_camera_board_intersection(self, camera_x):
        """Calculate where the camera epipolar line intersects the board surface.
        
        Args:
            camera_x: X position detected by camera in mm
            
        Returns:
            (x, y) position of intersection point on board surface
        """
        if camera_x is None:
            return None
            
        # Simple case - the X coordinate is directly from camera, Y is 0 (board surface)
        # In reality, we might need a more complex projection based on camera angle
        return (camera_x, 0)

    def calculate_final_tip_position(self, camera_point, lidar1_point, lidar2_point):
        """
        Calculate the final tip position using all available data with enhanced 3D lean correction.
        
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
        
        # Enhanced weighting system that considers forward/backward lean
        if camera_point is not None:
            if lidar1_point is not None and lidar2_point is not None:
                # Detect forward/backward lean
                forward_lean_angle, lean_confidence = self.detect_forward_backward_lean(lidar1_point, lidar2_point)
                
                # Significant lean detected with good confidence
                if abs(forward_lean_angle) > 5 and lean_confidence > 0.7:
                    # Direction of lean affects which LIDAR to trust more for Y position
                    if forward_lean_angle > 0:  # Leaning toward LIDAR 1
                        # Give more weight to LIDAR 1 for Y position
                        lidar_y = lidar1_point[1] * 0.7 + lidar2_point[1] * 0.3
                    else:  # Leaning toward LIDAR 2
                        # Give more weight to LIDAR 2 for Y position
                        lidar_y = lidar1_point[1] * 0.3 + lidar2_point[1] * 0.7
                else:
                    # No significant lean detected, use average of both LIDARs
                    lidar_y = (lidar1_point[1] + lidar2_point[1]) / 2
                    
                # Use camera for X position (more reliable)
                final_x = camera_point[0]
                final_y = lidar_y
                
                final_tip_position = (final_x, final_y)
            elif lidar1_point is not None:
                # Have camera and LIDAR 1
                final_tip_position = (camera_point[0], lidar1_point[1])
            elif lidar2_point is not None:
                # Have camera and LIDAR 2
                final_tip_position = (camera_point[0], lidar2_point[1])
            else:
                final_tip_position = camera_point
        
        # If only LIDARs are available (no camera)
        elif lidar1_point is not None and lidar2_point is not None:
            # Detect forward/backward lean to adjust weighting
            forward_lean_angle, lean_confidence = self.detect_forward_backward_lean(lidar1_point, lidar2_point)
            
            # If significant lean with good confidence
            if abs(forward_lean_angle) > 5 and lean_confidence > 0.7:
                if forward_lean_angle > 0:  # Leaning toward LIDAR 1
                    # Weight LIDAR 1 more for both X and Y
                    weight1 = 0.7
                    weight2 = 0.3
                else:  # Leaning toward LIDAR 2
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

    def update_lean_visualization(self, side_lean_angle, forward_lean_angle, lean_confidence):
        """Update the visualization of lean angles."""
        # Handle None values for lean angles
        if side_lean_angle is None:
            side_lean_angle = 90.0  # Default to vertical
        if forward_lean_angle is None:
            forward_lean_angle = 0.0
        if lean_confidence is None:
            lean_confidence = 0.0
            
        # Update text for lean angles
        self.lean_text.set_text(
            f"Side Lean: {side_lean_angle:.1f}° (90° = vertical)\n"
            f"Forward Lean: {forward_lean_angle:.1f}° (conf: {lean_confidence:.2f})"
        )
        
        # If we have a good forward lean detection, visualize it with an arrow
        if lean_confidence > 0.6 and abs(forward_lean_angle) > 5:
            # Create an arrow showing the forward lean direction
            # Arrow starts at origin (0,0)
            arrow_length = 50  # Length of arrow
            
            # Arrow direction depends on forward lean angle
            # Positive angle means leaning toward LIDAR 1 (left side)
            # Negative angle means leaning toward LIDAR 2 (right side)
            
            # Calculate arrow endpoint
            # If leaning toward LIDAR 1 (left), arrow points left and up
            # If leaning toward LIDAR 2 (right), arrow points right and up
            if forward_lean_angle > 0:
                # Leaning toward LIDAR 1 (left)
                arrow_dx = -arrow_length * np.sin(np.radians(forward_lean_angle))
                arrow_dy = arrow_length * np.cos(np.radians(forward_lean_angle))
            else:
                # Leaning toward LIDAR 2 (right)
                arrow_dx = arrow_length * np.sin(np.radians(-forward_lean_angle))
                arrow_dy = arrow_length * np.cos(np.radians(-forward_lean_angle))
                
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
                f"{abs(forward_lean_angle):.1f}°", 
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
        """Update plot data with enhanced 3D lean correction."""
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
                    self.lidar1_recent_points.append((x, y))  # FIXED: Changed from lidar2_recent_points
                    if len(self.lidar1_recent_points) > self.max_recent_points:
                        self.lidar1_recent_points.pop(0)
                    
                    x1.append(x)  # FIXED: Changed from x2
                    y1.append(y)  # FIXED: Changed from y2
                    
                    # Update most significant point (closest to center)
                    dist_from_center = np.sqrt(x**2 + y**2)
                    if lidar1_most_significant is None or dist_from_center < lidar1_most_significant[2]:
                        lidar1_most_significant = (x, y, dist_from_center)  # FIXED: Using lidar1_most_significant
                        
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
        camera_x = self.camera_data.get("dart_mm_x")
        side_lean_angle = self.camera_data.get("dart_angle", 90)  # Default to vertical if unknown
        
        # Calculate forward/backward lean angle using both LIDARs if available
        forward_lean_angle = 0
        lean_confidence = 0
        if lidar1_avg is not None and lidar2_avg is not None:
            forward_lean_angle, lean_confidence = self.detect_forward_backward_lean(lidar1_avg, lidar2_avg)
            
            # Add to lean history for smoothing
            self.lean_history.append((forward_lean_angle, lean_confidence))
            if len(self.lean_history) > self.max_lean_history:
                self.lean_history.pop(0)
                
            # Calculate weighted average of lean angles (higher confidence = higher weight)
            if self.lean_history:
                total_weight = sum(conf for _, conf in self.lean_history)
                if total_weight > 0:
                    smoothed_lean = sum(angle * conf for angle, conf in self.lean_history) / total_weight
                    forward_lean_angle = smoothed_lean
                    
            # Update current values for visualization
            self.current_forward_lean_angle = forward_lean_angle
            self.forward_lean_confidence = lean_confidence
        
        # Calculate intersection of camera vector with board surface
        self.camera_board_intersection = self.find_camera_board_intersection(camera_x)
        
        # Project LIDAR points accounting for both side-to-side and forward/backward lean
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None
        
        if lidar1_avg is not None:
            self.lidar1_projected_point = self.project_lidar_point_with_3d_lean(
                lidar1_avg, self.lidar1_height, side_lean_angle, forward_lean_angle, camera_x)
                
        if lidar2_avg is not None:
            self.lidar2_projected_point = self.project_lidar_point_with_3d_lean(
                lidar2_avg, self.lidar2_height, side_lean_angle, forward_lean_angle, camera_x)
        
        # If camera has no data, don't adjust LIDAR points for side-to-side lean
        # but still adjust for forward/backward lean if detected
        if camera_x is None and lidar1_avg is not None and lidar2_avg is not None:
            self.lidar1_projected_point = self.project_lidar_point_with_3d_lean(
                lidar1_avg, self.lidar1_height, 90, forward_lean_angle, None)
            self.lidar2_projected_point = self.project_lidar_point_with_3d_lean(
                lidar2_avg, self.lidar2_height, 90, forward_lean_angle, None)
        
        # Calculate final tip position using all available data with enhanced 3D lean correction
        final_tip_position = self.calculate_final_tip_position(
            self.camera_board_intersection, 
            self.lidar1_projected_point,
            self.lidar2_projected_point
        )
        
        # Apply calibration correction to final tip position
        if final_tip_position is not None:
            final_tip_position = self.apply_calibration_correction(
                final_tip_position[0], final_tip_position[1])
                
            # Also apply segment-specific coefficients
            final_tip_position = self.apply_segment_coefficients(
                final_tip_position[0], final_tip_position[1])
            
            # Update detected dart position
            self.detected_dart.set_data([final_tip_position[0]], [final_tip_position[1]])
            
            # Check which zone the dart is in
            distance_from_center = np.sqrt(final_tip_position[0]**2 + final_tip_position[1]**2)
            detected_zone = None
            for name, radius in self.radii.items():
                if distance_from_center <= radius:
                    detected_zone = name
                    break
            
            # Handle None values for lean angles before printing
            if side_lean_angle is None:
                side_lean_angle = 90.0  # Default to vertical
            if forward_lean_angle is None:
                forward_lean_angle = 0.0
            if lean_confidence is None:
                lean_confidence = 0.0
                    
            # Print detailed dart information with lean angles
            print(f"Dart detected - X: {final_tip_position[0]:.1f}, Y: {final_tip_position[1]:.1f}, "
                  f"Zone: {detected_zone if detected_zone else 'Outside'}")
            print(f"Side lean: {side_lean_angle:.1f}° (90° = vertical), "
                  f"Forward lean: {forward_lean_angle:.1f}° (conf: {lean_confidence:.2f})")
        else:
            self.detected_dart.set_data([], [])
        
        # Update the lean visualization
        self.update_lean_visualization(side_lean_angle, forward_lean_angle, lean_confidence)
            
        # Update camera visualization
        if self.camera_board_intersection is not None:
            # Calculate direction vector from camera to intersection
            camera_x = self.camera_board_intersection[0]
            camera_y = self.camera_board_intersection[1]
            
            # Calculate unit vector from camera to intersection
            dx = camera_x - self.camera_position[0]
            dy = camera_y - self.camera_position[1]
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize and scale to vector length
                unit_x = dx / length
                unit_y = dy / length
                vector_end_x = self.camera_position[0] + self.camera_vector_length * unit_x
                vector_end_y = self.camera_position[1] + self.camera_vector_length * unit_y
                
                # Draw camera vector and intersection point
                self.camera_vector.set_data(
                    [self.camera_position[0], vector_end_x],
                    [self.camera_position[1], vector_end_y]
                )
                self.camera_dart.set_data([camera_x], [camera_y])
            else:
                self.camera_vector.set_data([], [])
                self.camera_dart.set_data([], [])
        else:
            self.camera_vector.set_data([], [])
            self.camera_dart.set_data([], [])
            
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
                    [self.lidar1_pos[1], vector_end_y]
                )
                
                # Draw projected point if available
                if self.lidar1_projected_point is not None:
                    self.lidar1_dart.set_data(
                        [self.lidar1_projected_point[0]], 
                        [self.lidar1_projected_point[1]]
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
                    [self.lidar2_pos[1], vector_end_y]
                )
                
                # Draw projected point if available
                if self.lidar2_projected_point is not None:
                    self.lidar2_dart.set_data(
                        [self.lidar2_projected_point[0]], 
                        [self.lidar2_projected_point[1]]
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
        
        return (self.scatter1, self.scatter2, self.camera_vector, self.detected_dart,
                self.lidar1_vector, self.lidar2_vector, self.camera_dart, 
                self.lidar1_dart, self.lidar2_dart)
    def run(self, lidar1_script, lidar2_script):
        """Start all components."""
        lidar1_thread = threading.Thread(target=self.start_lidar, args=(lidar1_script, self.lidar1_queue, 1))
        lidar2_thread = threading.Thread(target=self.start_lidar, args=(lidar2_script, self.lidar2_queue, 2))
        camera_thread = threading.Thread(target=self.camera_detection)

        lidar1_thread.start()
        time.sleep(1)
        lidar2_thread.start()
        time.sleep(1)
        camera_thread.start()

        anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)
        plt.show()

        self.running = False
        lidar1_thread.join()
        lidar2_thread.join()
        camera_thread.join()

    def calibration_mode(self):
        """Interactive calibration for LIDAR rotation and coefficient scaling."""
        print("Calibration Mode")
        print("1. LIDAR Rotation Calibration")
        print("2. Coefficient Scaling Calibration")
        print("q. Quit")
        
        option = input("Select option: ")
        
        if option == "1":
            self._calibrate_lidar_rotation()
        elif option == "2":
            self._calibrate_coefficient_scaling()
        else:
            print("Exiting calibration mode.")
    
    def _calibrate_lidar_rotation(self):
        """Interactive calibration for LIDAR rotation."""
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
        """Interactive calibration for coefficient scaling factors."""
        print("Coefficient Scaling Calibration Mode")
        print("Adjust scaling factors for specific segments and ring types.")
        print("Format: [segment]:[ring_type]:[scale]")
        print("  - segment: 1-20 or 'all'")
        print("  - ring_type: 'doubles', 'trebles', 'small', 'large', or 'all'")
        print("  - scale: scaling factor (e.g. 0.5, 1.0, 1.5)")
        print("Example: 20:doubles:1.5 - Sets double ring scaling for segment 20 to 1.5")
        print("Example: all:trebles:0.8 - Sets treble ring scaling for all segments to 0.8")
        
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
                        self.coefficient_scaling[segment][rt] = scale
                        
                print(f"Updated scaling factors for {len(segments)} segment(s) and {len(ring_types)} ring type(s)")
                
                # Print current settings for verification
                if len(segments) <= 3:  # Only print details for a few segments to avoid cluttering
                    for segment in segments:
                        print(f"Segment {segment}: " + ", ".join([f"{rt}={self.coefficient_scaling[segment][rt]}" for rt in ring_types]))
                else:
                    # Just print a summary
                    print(f"Set {', '.join(ring_types)} scaling factor to {scale} for segments {segments[0]}-{segments[-1]}")
                    
            except ValueError:
                print("Scale must be a numeric value")
    
    def save_coefficient_scaling(self, filename="coefficient_scaling.json"):
        """Save the current coefficient scaling configuration to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.coefficient_scaling, f, indent=2)
            print(f"Coefficient scaling saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving coefficient scaling: {e}")
            return False
    
    def load_coefficient_scaling(self, filename="coefficient_scaling.json"):
        """Load coefficient scaling configuration from a JSON file."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_scaling = json.load(f)
                    
                # Convert string keys back to integers
                self.coefficient_scaling = {int(k): v for k, v in loaded_scaling.items()}
                print(f"Coefficient scaling loaded from {filename}")
                return True
            else:
                print(f"Scaling file {filename} not found, using defaults")
                return False
        except Exception as e:
            print(f"Error loading coefficient scaling: {e}")
            return False

if __name__ == "__main__":
    lidar1_script = "./tri_test_lidar1"
    lidar2_script = "./tri_test_lidar2"
    visualizer = LidarCameraVisualizer()
    
    # Try to load coefficient scaling from file
    visualizer.load_coefficient_scaling()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            visualizer.calibration_mode()
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
