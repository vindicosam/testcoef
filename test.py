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
        self.lidar1_rotation = 342        # Adjusted angle
        self.lidar2_rotation = 187.25       # Adjusted angle
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
        self.camera_data = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}  # Now tracking tip pixel too

        # ROI Settings for side camera (similar to camera2 in the dual setup)
        self.camera_board_plane_y = 228  # The y-coordinate where the board surface is
        self.camera_roi_range = 15      # How much above and below to include
        self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
        self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
        self.camera_roi_left = 117   # Example left boundary
        self.camera_roi_right = 604  # Example right boundary
        
        # Add calibration point system
        self.camera_calibration_points = []  # List of (pixel_x, mm_y) tuples for calibration
        # Default linear calibration as fallback
        self.pixel_to_mm_factor = -0.628  # Slope in mm/pixel 
        self.pixel_offset = 192.8        # Board y when pixel_x = 0

        # Detection persistence to maintain visibility
        self.last_valid_detection = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 5000

        # Camera background subtractor with improved settings from second script
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=70,
            detectShadows=False
        )
        
        # Previous frame storage for frame differencing (from second script)
        self.prev_gray = None
        
        # Frame differencing threshold from second script
        self.diff_threshold = 67
        self.min_contour_area = 67

        # LIDAR queues
        self.lidar1_queue = Queue()
        self.lidar2_queue = Queue()
        
        # Storage for most recent LIDAR data points
        self.lidar1_recent_points = []
        self.lidar2_recent_points = []
        self.max_recent_points = 50  # Keep last 20 points for smoothing

        # Store the projected LIDAR points (after lean compensation)
        self.lidar1_projected_point = None
        self.lidar2_projected_point = None
        
        # Intersection point of camera vector with board plane
        self.camera_board_intersection = None

        # CSV logging initialization
        self.initialize_csv_logging()

        # Dartboard scoring configuration (for display purposes)
        self.score_text = None  # Will hold a text element to display the score

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

        # NEW: Add a dictionary to store per-segment radial offsets (toward or away from bullseye)
        self.segment_radial_offsets = {}
        # Initialize with 0 offset for all segments
        for segment in range(1, 21):
            self.segment_radial_offsets[segment] = -15  # mm, positive = away from bull, negative = toward bull

        # All coefficient dictionaries included to ensure they exist
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
            "3_3": {"x_correction": -12.353, "y_correction": 8.187}
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
            "4_5": {"x_correction": -2.475, "y_correction": 6.135}
        }

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
            (-23, -167): (0.4, 0.6),  # Singles area (outer)
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
        
        # Setup visualization
        self.setup_plot()

        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
	    
    def load_standard_calibration_points(self):
	    """Load standard set of calibration points mapping pixel X to mm Y."""
	    # Clear existing calibration points
	    self.clear_calibration_points()
	    
	    calibration_points = [
	        (142, 174),    # 20 OC
	        (552, -174),   # 3 OC
	        (412, 0),      # 11 OC
	        (340, 0),      # 6 OC
	        (172, 260),    # 20 TSTR
	        (220, 107),    # 20 TSBL
	        (243, 97),     # 20 TBR
	        (335, 15),     # 20 BSBC
	        (205, 143),    # 1 TSTR
	        (231, 105),    # 1 TSBL
	        (258, 88),     # 1 TBR
	        (337, 14),     # 1 BSBC
	        (243, 113),    # 18 TSTR
	        (248, 94),     # 18 TSBL
	        (276, 69),     # 18 TBR
	        (338, 13),     # 18 BSBC
	        (282, 73),     # 4 TSTR
	        (274, 76),     # 4 TSBL
	        (301, 46),     # 4 TBR
	        (343, 11),     # 4 BSBC
	        (318, 27),     # 13 TSTR
	        (302, 47),     # 13 TSBL
	        (330, 19),     # 13 TBR
	        (348, 3),      # 13 BSBC
	        (358, -23),    # 6 TSTR
	        (330, 15),     # 6 TSBL
	        (357, -14),    # 6 TBR
	        (353, 0),      # 6 BSBC
	        (398, -71),    # 10 TSTR
	        (361, -19),    # 10 TSBL
	        (384, -44),    # 10 TBR
	        (359, -5),     # 10 BSBC
	        (435, -113),   # 15 TSTR
	        (392, -50),    # 15 TSBL
	        (412, -67),    # 15 TBR
	        (366, -11),    # 15 BSBC
	        (478, -140),   # 2 TSTR
	        (420, -78),    # 2 TSBL
	        (438, -85),    # 2 TBR
	        (370, -14),    # 2 BSBC
	        (515, -158),   # 17 TSTR
	        (446, -97),    # 17 TSBL
	        (459, -97),    # 17 TBR
	        (374, -16),    # 17 BSBC
	        (549, -157),   # 3 TSTR
	        (470, -108),   # 3 TSBL
	        (474, -97),    # 3 TBR
	        (378, -17),    # 3 BSBC
	        (574, -142),   # 19 TSTR
	        (487, -107),   # 19 TSBL
	        (481, -90),    # 19 TBR
	        (377, -17),    # 19 BSBC
	        (574, -115),   # 7 TSTR
	        (493, -96),    # 7 TSBL
	        (472, -69),    # 7 TBR
	        (375, -14),    # 7 BSBC
	        (540, -74),    # 16 TSTR
	        (485, -75),    # 16 TSBL
	        (446, -45),    # 16 TBR
	        (371, -11),    # 16 BSBC
	        (458, -27),    # 8 TSTR
	        (454, -47),    # 8 TSBL
	        (403, -17),    # 8 TBR
	        (366, -5),     # 8 BSBC
	        (347, 23),     # 11 TSTR
	        (405, -24),    # 11 TSBL
	        (352, 14),     # 11 TBR
	        (359, 0),      # 11 BSBC
	        (241, 71),     # 14 TSTR
	        (347, 19),     # 14 TSBL
	        (303, 45),     # 14 TBR
	        (352, 6),      # 14 BSBC
	        (171, 112),    # 9 TSTR
	        (293, 51),     # 9 TSBL
	        (347, 69),     # 9 TBR
	        (345, 12),     # 9 BSBC
	        (144, 142),    # 12 TSTR
	        (246, 77),     # 12 TSBL
	        (240, 85),     # 12 TBR
	        (342, 13),     # 12 BSBC
	        (147, 158),    # 5 TSTR
	        (224, 95),     # 5 TSBL
	        (234, 108),    # 5 TBR
	        (336, 19)      # 5 BSBC
	    ]
	    
	    # Add all the calibration points
	    for pixel_x, mm_y in calibration_points:
	        self.add_calibration_point(pixel_x, mm_y)
	    
	    print(f"Loaded {len(calibration_points)} standard calibration points")	
    def xy_to_dartboard_score(self, x, y):
        """
        Convert x,y coordinates to dartboard score.
        
        Args:
            x (float): X coordinate in mm, with 0 at the center of the dartboard
            y (float): Y coordinate in mm, with 0 at the center of the dartboard
            
        Returns:
            str: Dartboard score notation (e.g. "T20", "D16", "S7", "B", "OB", "Outside")
        """
        # Dartboard scores configuration (clockwise from right of 20)
        scores = [13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6]
        
        # Calculate distance from center and angle
        distance = np.sqrt(x * x + y * y)
        angle = np.degrees(np.arctan2(y, x))
        
        # Adjust angle to match dartboard orientation (20 at top)
        angle = (angle - 9 + 360) % 360
        
        # Determine score based on angle
        segment_idx = int(angle / 18)
        if segment_idx >= len(scores):  # Safety check
            return "Outside"
            
        base_score = scores[segment_idx]
        
        # Determine multiplier and code based on distance
        if distance <= self.radii['bullseye']:
            return "B"  # Bullseye (50)
        elif distance <= self.radii['outer_bull']:
            return "OB"  # Outer bull (25)
        elif self.radii['inner_treble'] < distance <= self.radii['outer_treble']:
            return f"T{base_score}"  # Triple
        elif self.radii['inner_double'] < distance <= self.radii['outer_double']:
            return f"D{base_score}"  # Double
        elif distance <= self.radii['outer_double']:
            return f"S{base_score}"  # Single
        else:
            return "Outside"

    def get_score_description(self, score):
        """Return a human-readable description of the score."""
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
        """Initialize CSV file for logging dart data."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"dart_data_{timestamp}.csv"
        
        # Create CSV file with headers
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Timestamp', 'Dart_X_mm', 'Dart_Y_mm', 
                                'Tip_Pixel_X', 'Tip_Pixel_Y', 
                                'Side_Lean_Angle', 'Up_Down_Lean_Angle',
                                'Score'])
        
        print(f"CSV logging initialized: {self.csv_filename}")

    def log_dart_data(self, final_tip_position, tip_pixel, side_lean_angle, up_down_lean_angle):
        """Log dart data to CSV file."""
        if final_tip_position is None:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Calculate score if we have a valid position
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
            angle: Angle in degrees (90ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â° = vertical) or None if angle couldn't be calculated
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
            self.camera_data["tip_pixel"] = None

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
        
        # Determine which segment we're in
        angle_deg = math.degrees(math.atan2(y, x))
        if angle_deg < 0:
            angle_deg += 360
        segment = int(((angle_deg + 9) % 360) / 18) + 1
        
        # Apply radial offset (if any) for this segment
        radial_offset = self.segment_radial_offsets.get(segment, 0.0)
        if radial_offset != 0.0:
            # Calculate direction vector from center to point
            magnitude = math.sqrt(x*x + y*y)
            if magnitude > 0:  # Avoid division by zero
                # Unit vector in direction from center to point
                unit_x = x / magnitude
                unit_y = y / magnitude
                # Apply offset in this direction (negative = toward bull, positive = away from bull)
                x += unit_x * radial_offset
                y += unit_y * radial_offset
        
        # If far from any wire, return the point with only radial offset applied
        if wire_factor <= 0.0:
            return x, y
        
        dist = math.sqrt(x*x + y*y)
        
        # No correction for bullseye and outer bull
        if dist <= self.radii["outer_bull"]:
            return x, y

        # Determine segment number from angle (using offset of 9ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â° as before)
        # We recalculate after applying the radial offset
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
            lean_angle: Estimated lean angle in degrees (0ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â° = vertical, positive = up, negative = down)
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
        Enhanced version with better handling of lean directions.
        
        Args:
            lidar_point: (x, y) position of LIDAR detection
            lidar_height: height of the LIDAR beam above board in mm
            side_lean_angle: angle of dart from vertical in degrees (90Ãƒâ€šÃ‚Â° = vertical, from camera)
            up_down_lean_angle: angle of up/down lean in degrees (0Ãƒâ€šÃ‚Â° = vertical)
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
            # Determine lean direction from angle
            # A left lean (angle < 85) should move Y coordinate DOWN
            # A right lean (angle > 95) should move Y coordinate UP
            lean_direction = 0  # No lean correction by default
            
            if side_lean_angle < 85:
                # Left lean
                lean_factor = (85 - side_lean_angle) / 85.0  # More horizontal = higher factor
                lean_direction = -1  # Move downward
            elif side_lean_angle > 95:
                # Right lean
                lean_factor = (side_lean_angle - 95) / 85.0  # More horizontal = higher factor
                lean_direction = 1  # Move upward
            else:
                # Near vertical, little to no correction
                lean_factor = 0
            
            # Calculate Y displacement (how far LIDAR point is from camera line)
            y_displacement = abs(original_y - camera_y)
            
            # Apply side-to-side adjustment proportional to lean angle and displacement
            MAX_SIDE_ADJUSTMENT = self.side_lean_max_adjustment  # mm
            
            # The further from the camera line, the greater the correction
            side_adjustment = min(lean_factor * y_displacement, MAX_SIDE_ADJUSTMENT)
            
            # Apply in the appropriate direction
            adjusted_y = original_y + (side_adjustment * lean_direction)
        
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
        up_down_adjustment = (up_down_lean_angle / self.MAX_UP_DOWN_LEAN) * (y_distance_from_center / 170.0) * MAX_UP_DOWN_ADJUSTMENT
        
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
        Calculate the final tip position using LIDAR data only, with the camera used just for lean detection.
        
        Args:
            camera_point: Intersection of camera vector with board (used only for lean detection, not position)
            lidar1_point: Projected LIDAR 1 point
            lidar2_point: Projected LIDAR 2 point
            
        Returns:
            (x, y) final estimated tip position
        """
        # Points that are actually available
        valid_lidar_points = []
        if lidar1_point is not None:
            valid_lidar_points.append(lidar1_point)
        if lidar2_point is not None:
            valid_lidar_points.append(lidar2_point)
            
        # If no valid LIDAR points, return None - we can't determine position reliably
        # Camera is now used only for lean detection, not for positioning
        if not valid_lidar_points:
            return None
            
        # If only one LIDAR has data, use that LIDAR for positioning
        if len(valid_lidar_points) == 1:
            # Get the LIDAR point's original coordinates
            x, y = valid_lidar_points[0]
            
            # Apply side lean correction if available, but keep the LIDAR y-coordinate
            if self.camera_data["dart_angle"] is not None:
                side_lean_angle = self.camera_data["dart_angle"]
                
                # Apply lean-based Y correction if significant lean detected
                if side_lean_angle < 85:  # Left lean
                    # Calculate correction factor based on how much lean (more horizontal = more correction)
                    lean_factor = (85 - side_lean_angle) / 85.0
                    # Move Y coordinate DOWN for left lean
                    y_correction = -lean_factor * self.side_lean_max_adjustment
                    y += y_correction
                elif side_lean_angle > 95:  # Right lean
                    # Calculate correction factor based on how much lean
                    lean_factor = (side_lean_angle - 95) / 85.0
                    # Move Y coordinate UP for right lean
                    y_correction = lean_factor * self.side_lean_max_adjustment
                    y += y_correction
            
            # Apply scale correction
            x = x * self.x_scale_correction
            y = y * self.y_scale_correction
            
            return (x, y)
        
        # When both LIDARs have data, we can detect up/down lean and weight accordingly
        up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        
        # Enhanced weighting system that considers up/down lean
        if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
            # Direction of lean affects which LIDAR to trust more
            if up_down_lean_angle > 0:  # Leaning upward
                # Give more weight to LIDAR 1
                weight1 = 0.7
                weight2 = 0.3
            else:  # Leaning downward
                # Give more weight to LIDAR 2
                weight1 = 0.3
                weight2 = 0.7
                
            # Calculate weighted average of LIDAR positions
            x = lidar1_point[0] * weight1 + lidar2_point[0] * weight2
            y = lidar1_point[1] * weight1 + lidar2_point[1] * weight2
        else:
            # No significant up/down lean, use equal weighting for LIDARs
            x = (lidar1_point[0] + lidar2_point[0]) / 2
            y = (lidar1_point[1] + lidar2_point[1]) / 2
            
        # Apply additional side lean correction if available
        if self.camera_data["dart_angle"] is not None:
            side_lean_angle = self.camera_data["dart_angle"]
            
            # Apply lean-based Y correction only if significant lean detected
            if side_lean_angle < 85:  # Left lean
                # Calculate correction factor based on how much lean (more horizontal = more correction)
                lean_factor = (85 - side_lean_angle) / 85.0
                # Move Y coordinate DOWN for left lean
                y_correction = -lean_factor * self.side_lean_max_adjustment
                y += y_correction
            elif side_lean_angle > 95:  # Right lean
                # Calculate correction factor based on how much lean
                lean_factor = (side_lean_angle - 95) / 85.0
                # Move Y coordinate UP for right lean
                y_correction = lean_factor * self.side_lean_max_adjustment
                y += y_correction
        
        # Apply scale correction to final position
        x = x * self.x_scale_correction
        y = y * self.y_scale_correction
        
        return (x, y)

    def update_lean_visualization(self, side_lean_angle, up_down_lean_angle, lean_confidence):
        """Update the visualization of lean angles."""
        # Update or create the arrow for visualization
        arrow_length = 40
        arrow_x = -350
        arrow_y = 350
        
        # Calculate arrow components based on lean angles
        if side_lean_angle is not None and up_down_lean_angle is not None:
            # Convert from degrees to radians for math functions
            side_lean_rad = np.radians(90 - side_lean_angle)  # 90ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â° is vertical
            up_down_lean_rad = np.radians(up_down_lean_angle)
            
            # X component affected by both side and up/down lean
            dx = arrow_length * np.sin(side_lean_rad) * np.cos(up_down_lean_rad)
            # Y component affected by side lean
            dy = arrow_length * np.cos(side_lean_rad)
            # Z component (into screen) affected by up/down lean - not directly shown in 2D
            
            # Remove old arrow if exists
            if self.lean_arrow:
                self.lean_arrow.remove()
            
            # Create new arrow
            self.lean_arrow = self.ax.arrow(
                arrow_x, arrow_y, dx, dy, 
                head_width=8, head_length=10, 
                fc='purple', ec='purple', alpha=0.7,
                length_includes_head=True
            )
            
            # Add confidence level to visualization
            if self.arrow_text:
                self.arrow_text.remove()
            
            self.arrow_text = self.ax.text(
                arrow_x + dx + 10, arrow_y + dy, 
                f"Conf: {lean_confidence:.2f}", 
                fontsize=8, color='purple'
            )
        
        # If no data, show vertical arrow
        elif self.lean_arrow is None:
            self.lean_arrow = self.ax.arrow(
                arrow_x, arrow_y, 0, arrow_length, 
                head_width=8, head_length=10, 
                fc='gray', ec='gray', alpha=0.5,
                length_includes_head=True
            )

    def update_plot(self, frame):
        """Update plot data with enhanced 3D lean correction and CSV logging."""
        # Process LIDAR data
        lidar1_points_x = []
        lidar1_points_y = []
        lidar2_points_x = []
        lidar2_points_y = []
        
        # Process LIDAR 1 queue
        while not self.lidar1_queue.empty():
            angle, distance = self.lidar1_queue.get()
            x, y = self.polar_to_cartesian(angle, distance, self.lidar1_pos, 
                                        self.lidar1_rotation, self.lidar1_mirror)
            if x is not None and y is not None:
                # Filter points by radii
                in_range, _ = self.filter_points_by_radii(x, y)
                if in_range:
                    lidar1_points_x.append(x)
                    lidar1_points_y.append(y)
                    self.lidar1_recent_points.append((x, y))
        
        # Process LIDAR 2 queue
        while not self.lidar2_queue.empty():
            angle, distance = self.lidar2_queue.get()
            x, y = self.polar_to_cartesian(angle, distance, self.lidar2_pos, 
                                        self.lidar2_rotation, self.lidar2_mirror)
            if x is not None and y is not None:
                # Filter points by radii
                in_range, _ = self.filter_points_by_radii(x, y)
                if in_range:
                    lidar2_points_x.append(x)
                    lidar2_points_y.append(y)
                    self.lidar2_recent_points.append((x, y))
        
        # Keep only the most recent points
        self.lidar1_recent_points = self.lidar1_recent_points[-self.max_recent_points:]
        self.lidar2_recent_points = self.lidar2_recent_points[-self.max_recent_points:]
        
        # Get the camera data
        camera_y = self.camera_data["dart_mm_y"]
        side_lean_angle = self.camera_data["dart_angle"]
        
        # Calculate lean angle
        up_down_lean_angle = 0
        lean_confidence = 0
        
        if len(self.lidar1_recent_points) > 0 and len(self.lidar2_recent_points) > 0:
            lidar1_point = self.lidar1_recent_points[-1]
            lidar2_point = self.lidar2_recent_points[-1]
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        
        # Update lean visualization
        self.update_lean_visualization(side_lean_angle, up_down_lean_angle, lean_confidence)
        
        # Find where the camera vector intersects with the board
        camera_point = self.find_camera_board_intersection(camera_y)
        
        # Project LIDAR points with 3D lean correction
        lidar1_projected = None
        lidar2_projected = None
        
        if len(self.lidar1_recent_points) > 0:
            lidar1_point = self.lidar1_recent_points[-1]
            lidar1_projected = self.project_lidar_point_with_3d_lean(
                lidar1_point, self.lidar1_height, side_lean_angle, 
                up_down_lean_angle, camera_y
            )
        
        if len(self.lidar2_recent_points) > 0:
            lidar2_point = self.lidar2_recent_points[-1]
            lidar2_projected = self.project_lidar_point_with_3d_lean(
                lidar2_point, self.lidar2_height, side_lean_angle, 
                up_down_lean_angle, camera_y
            )
        
        # Calculate final tip position
        final_tip_position = self.calculate_final_tip_position(
            camera_point, lidar1_projected, lidar2_projected
        )
        
        # Apply segment-specific coefficients
        if final_tip_position is not None:
            x, y = final_tip_position
            x, y = self.apply_segment_coefficients(x, y)
            x, y = self.apply_calibration_correction(x, y)
            final_tip_position = (x, y)
        
        # Log data to CSV
        self.log_dart_data(
            final_tip_position, 
            self.camera_data["tip_pixel"], 
            side_lean_angle, 
            up_down_lean_angle
        )
        
        # Update plot with new data
        self.scatter1.set_data(lidar1_points_x, lidar1_points_y)
        self.scatter2.set_data(lidar2_points_x, lidar2_points_y)
        
        # Update camera vector
        if camera_point is not None:
            # Calculate a point extending beyond the camera_point by the vector_length
            # Since camera is on the left (-X axis), the vector extends toward +X
            board_x = 0  # X coordinate of the board plane
            
            # Direction vector pointing from camera to board intersection
            dir_x = board_x - self.camera_position[0]  # Positive value
            dir_y = camera_point[1] - self.camera_position[1]
            
            # Normalize the direction vector
            vector_length = np.sqrt(dir_x**2 + dir_y**2)
            if vector_length > 0:
                norm_dir_x = dir_x / vector_length
                norm_dir_y = dir_y / vector_length
            else:
                norm_dir_x, norm_dir_y = 1, 0  # Default to horizontal if zero length
            
            # Calculate extended vector that goes well beyond the board
            extended_x = self.camera_position[0] + norm_dir_x * self.camera_vector_length
            extended_y = self.camera_position[1] + norm_dir_y * self.camera_vector_length
            
            # Update the camera vector to go through the point and extend beyond
            self.camera_vector.set_data(
                [self.camera_position[0], extended_x],
                [self.camera_position[1], extended_y]
            )
            self.camera_dart.set_data([camera_point[0]], [camera_point[1]])
        else:
            self.camera_vector.set_data([], [])
            self.camera_dart.set_data([], [])
        
        # Update LIDAR projections
        if lidar1_projected is not None:
            self.lidar1_dart.set_data([lidar1_projected[0]], [lidar1_projected[1]])
        else:
            self.lidar1_dart.set_data([], [])
            
        if lidar2_projected is not None:
            self.lidar2_dart.set_data([lidar2_projected[0]], [lidar2_projected[1]])
        else:
            self.lidar2_dart.set_data([], [])
        
        # Update final tip position
        if final_tip_position is not None:
            self.detected_dart.set_data([final_tip_position[0]], [final_tip_position[1]])
            
            # Update score text if point is on board
            score = self.xy_to_dartboard_score(final_tip_position[0], final_tip_position[1])
            if score != "Outside":
                description = self.get_score_description(score)
                if self.score_text:
                    self.score_text.set_text(description)
                else:
                    self.score_text = self.ax.text(-380, 360, description, fontsize=12, color='red')
        else:
            self.detected_dart.set_data([], [])
        
        # Update lean text
        # Properly handle None values in f-string formatting
        if side_lean_angle is not None:
            side_lean_str = f"{side_lean_angle:.1f}Ãƒâ€šÃ‚Â°"  # Fixed degree symbol
        else:
            side_lean_str = "N/A"

        if up_down_lean_angle is not None:
            up_down_lean_str = f"{up_down_lean_angle:.1f}Ãƒâ€šÃ‚Â°"  # Fixed degree symbol
        else:
            up_down_lean_str = "N/A"

        lean_text = f"Side Lean: {side_lean_str}\nUp/Down: {up_down_lean_str}"
        self.lean_text.set_text(lean_text)
        
        # Return all the artists that need to be redrawn
        artists = [
            self.scatter1, self.scatter2, 
            self.camera_vector, self.camera_dart,
            self.lidar1_dart, self.lidar2_dart, 
            self.detected_dart, self.lean_text
        ]
        
        if hasattr(self, 'score_text') and self.score_text:
            artists.append(self.score_text)
        
        if hasattr(self, 'lean_arrow') and self.lean_arrow:
            artists.append(self.lean_arrow)
        
        if hasattr(self, 'arrow_text') and self.arrow_text:
            artists.append(self.arrow_text)
            
        return artists
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
        valid_lidar_points = []
        if lidar1_point is not None:
            valid_lidar_points.append(lidar1_point)
        if lidar2_point is not None:
            valid_lidar_points.append(lidar2_point)
            
        # If no valid LIDAR points, can't determine position reliably
        # Camera is now used primarily for lean detection, not for positioning
        if not valid_lidar_points:
            return camera_point
            
        # If only one LIDAR has data, use that LIDAR for positioning
        if len(valid_lidar_points) == 1:
            # Apply the camera's Y position if available
            if camera_point is not None and self.camera_data["dart_angle"] is not None:
                # Determine lean direction and apply correction
                side_lean_angle = self.camera_data["dart_angle"]
                
                    # Get the LIDAR point's original coordinates
                x, y = valid_lidar_points[0]
                
                # Apply lean-based Y correction if significant lean detected
                if side_lean_angle < 85:  # Left lean
                    # Calculate correction factor based on how much lean (more horizontal = more correction)
                    lean_factor = (85 - side_lean_angle) / 85.0
                    # Move Y coordinate DOWN for left lean
                    y_correction = -lean_factor * self.side_lean_max_adjustment
                    y += y_correction
                elif side_lean_angle > 95:  # Right lean
                    # Calculate correction factor based on how much lean
                    lean_factor = (side_lean_angle - 95) / 85.0
                    # Move Y coordinate UP for right lean
                    y_correction = lean_factor * self.side_lean_max_adjustment
                    y += y_correction
                
                # Apply scale correction
                x = x * self.x_scale_correction
                y = y * self.y_scale_correction
                
                return (x, camera_point[1])
            else:
                # Just use the LIDAR point with scale correction
                x, y = valid_lidar_points[0]
                x = x * self.x_scale_correction
                y = y * self.y_scale_correction
                return (x, y)
        
        # When both LIDARs have data, we can detect up/down lean and weight accordingly
        up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        
        # Enhanced weighting system that considers up/down lean
        if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
            # Direction of lean affects which LIDAR to trust more
            if up_down_lean_angle > 0:  # Leaning upward
                # Give more weight to LIDAR 1
                weight1 = 0.7
                weight2 = 0.3
            else:  # Leaning downward
                # Give more weight to LIDAR 2
                weight1 = 0.3
                weight2 = 0.7
                
            # Calculate weighted average of LIDAR positions
            x = lidar1_point[0] * weight1 + lidar2_point[0] * weight2
            y = lidar1_point[1] * weight1 + lidar2_point[1] * weight2
        else:
            # No significant up/down lean, use equal weighting for LIDARs
            x = (lidar1_point[0] + lidar2_point[0]) / 2
            y = (lidar1_point[1] + lidar2_point[1]) / 2
            
        # If camera data is available, apply additional side lean correction
        if camera_point is not None and self.camera_data["dart_angle"] is not None:
            side_lean_angle = self.camera_data["dart_angle"]
            
            # Apply lean-based Y correction only if significant lean detected
            if side_lean_angle < 85:  # Left lean
                # Calculate correction factor based on how much lean (more horizontal = more correction)
                lean_factor = (85 - side_lean_angle) / 85.0
                # Move Y coordinate DOWN for left lean
                y_correction = -lean_factor * self.side_lean_max_adjustment
                y += y_correction
            elif side_lean_angle > 95:  # Right lean
                # Calculate correction factor based on how much lean
                lean_factor = (side_lean_angle - 95) / 85.0
                # Move Y coordinate UP for right lean
                y_correction = lean_factor * self.side_lean_max_adjustment
                y += y_correction
        
        # Apply scale correction to final position
        x = x * self.x_scale_correction
        y = y * self.y_scale_correction
        
        return (x, y)    
    def run(self, lidar1_script, lidar2_script):
        """Start all components with the specified LIDAR scripts."""
        # Start background threads
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
        
        # Start with delays between components - these are crucial!
        lidar1_thread.start()
        time.sleep(1)  # Give LIDAR 1 time to initialize
        
        lidar2_thread.start()
        time.sleep(1)  # Give LIDAR 2 time to initialize
        
        camera_thread.start()
        
        # Start animation
        self.ani = FuncAnimation(
            self.fig, self.update_plot, 
            blit=True, interval=100, 
            cache_frame_data=False
        )
        
        plt.show()
        
    # NEW: Method to set segment radial offset
    def set_segment_radial_offset(self, segment, offset_mm):
        """
        Set a radial offset for a specific segment.
        
        Args:
            segment (int): Segment number (1-20)
            offset_mm (float): Offset in mm (positive = away from bull, negative = toward bull)
        """
        if 1 <= segment <= 20:
            self.segment_radial_offsets[segment] = offset_mm
            print(f"Set segment {segment} radial offset to {offset_mm} mm")
        else:
            print(f"Invalid segment number: {segment}. Must be between 1-20.")
            
    # NEW: Method to get segment radial offset
    def get_segment_radial_offset(self, segment):
        """
        Get the radial offset for a specific segment.
        
        Args:
            segment (int): Segment number (1-20)
            
        Returns:
            float: Offset in mm
        """
        if 1 <= segment <= 20:
            return self.segment_radial_offsets[segment]
        else:
            print(f"Invalid segment number: {segment}. Must be between 1-20.")
            return 0.0
            
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

    def load_segment_radial_offsets(self, filename="segment_offsets.json"):
        """Load segment radial offsets from a JSON file."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_offsets = json.load(f)
                    
                # Convert string keys back to integers
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
        """Save the current segment radial offsets to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.segment_radial_offsets, f, indent=2)
            print(f"Segment radial offsets saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving segment radial offsets: {e}")
            return False
        
    # Calibration mode methods (added from second script)
    def calibration_mode(self):
        """Interactive calibration for LIDAR rotation and coefficient scaling."""
        print("Calibration Mode")
        print("1. LIDAR Rotation Calibration")
        print("2. Coefficient Scaling Calibration")
        print("3. Segment Radial Offset Calibration")  # NEW: Option for segment offset calibration
        print("q. Quit")
        
        option = input("Select option: ")
        
        if option == "1":
            self._calibrate_lidar_rotation()
        elif option == "2":
            self._calibrate_coefficient_scaling()
        elif option == "3":
            self._calibrate_segment_radial_offsets()  # NEW: Segment offset calibration
        else:
            print("Exiting calibration mode.")
    
    def _calibrate_lidar_rotation(self):
        """Interactive calibration for LIDAR rotation."""
        print("LIDAR Rotation Calibration Mode")
        print(f"Current LIDAR1 rotation: {self.lidar1_rotation}ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
        print(f"Current LIDAR2 rotation: {self.lidar2_rotation}ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
        
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
                    
                print(f"Updated LIDAR1 rotation: {self.lidar1_rotation}ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
                print(f"Updated LIDAR2 rotation: {self.lidar2_rotation}ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â°")
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
                
    # NEW: Segment radial offset calibration
    def _calibrate_segment_radial_offsets(self):
        """Interactive calibration for segment radial offsets."""
        print("Segment Radial Offset Calibration Mode")
        print("Current segment radial offsets (mm):")
        
        # Display current offsets
        for segment in range(1, 21):
            offset = self.segment_radial_offsets[segment]
            direction = "toward bull" if offset < 0 else "away from bull"
            if offset == 0:
                direction = "no offset"
            print(f"Segment {segment}: {abs(offset):.1f} mm {direction}")
        
        print("\nEnter commands in format: [segment]:[offset] ")
        print("  - segment: 1-20 or 'all'")
        print("  - offset: value in mm (positive = away from bull, negative = toward bull)")
        print("Example: 3:11 - Moves segment 3 11mm away from bull")
        print("Example: 6:-5 - Moves segment 6 5mm toward bull")
        print("Example: all:0 - Resets all segments to no offset")
        
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
                
                # Process segment specification
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
                
                # Update offsets
                for segment in segments:
                    self.segment_radial_offsets[segment] = offset
                    
                direction = "toward bull" if offset < 0 else "away from bull"
                if offset == 0:
                    direction = "no offset"
                    
                print(f"Updated offset for {len(segments)} segment(s) to {abs(offset):.1f} mm {direction}")
                    
            except ValueError:
                print("Offset must be a numeric value")
if __name__ == "__main__":
    # Use same LIDAR script paths as in your original code
    lidar1_script = "./tri_test_lidar1"
    lidar2_script = "./tri_test_lidar2"
    
    visualizer = LidarCameraVisualizer()
    visualizer.load_standard_calibration_points()
    
    # Try to load settings from files
    visualizer.load_coefficient_scaling()
    visualizer.load_segment_radial_offsets()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            visualizer.calibration_mode()
            # After calibration, ask to save settings
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
