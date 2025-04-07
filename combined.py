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
from bisect import bisect_left

class LidarDualCameraVisualizer:
    def __init__(self, cam_index1=0, cam_index2=2):
        # Store camera indices just like in script 2
        self.cam_index1 = cam_index1  # Front camera
        self.cam_index2 = cam_index2  # Side camera
        
        # =================== LIDAR SYSTEM ======================
        # Fixed LIDAR positions relative to the dartboard
        self.lidar1_pos = (-202.5, 224.0)  # Adjusted based on calibration
        self.lidar2_pos = (204.0, 223.5)   # Adjusted based on calibration

        # LIDAR configurations
        self.lidar1_rotation = 342        # Adjusted angle
        self.lidar2_rotation = 187.25     # Adjusted angle
        self.lidar1_mirror = True
        self.lidar2_mirror = True
        self.lidar1_offset = 4.5          # Adjusted offset
        self.lidar2_offset = 0.5          # Adjusted offset

        # LIDAR heights above board surface
        self.lidar1_height = 4.0  # mm above board surface
        self.lidar2_height = 8.0  # mm above board surface

        # Side camera configuration
        self.side_camera_position = (-350, 0)     # Camera is to the left of the board
        self.side_camera_vector_length = 1600     # Vector length in mm
        self.side_camera_data = {"dart_mm_y": None, "dart_angle": None, "tip_pixel": None}

        # ROI Settings for side camera (similar to camera2 in the dual setup)
        self.side_camera_board_plane_y = 228  # The y-coordinate where the board surface is
        self.side_camera_roi_range = 20        # How much above and below to include
        self.side_camera_roi_top = self.side_camera_board_plane_y - self.side_camera_roi_range
        self.side_camera_roi_bottom = self.side_camera_board_plane_y + self.side_camera_roi_range
        self.side_camera_roi_left = 117       # Left boundary
        self.side_camera_roi_right = 604      # Right boundary
        
        # Side camera background subtractor
        self.side_camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=70,
            detectShadows=False
        )
        
        # Previous frame storage for frame differencing
        self.prev_gray = None
        
        # Frame differencing threshold
        self.diff_threshold = 67
        self.min_contour_area = 67

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
        
        # Side camera intersection with board plane
        self.side_camera_board_intersection = None
        
        # 3D lean detection variables for LIDAR system
        self.current_up_down_lean_angle = 0.0
        self.up_down_lean_confidence = 0.0
        self.lean_history = []
        self.max_lean_history = 60
        self.lean_arrow = None
        self.arrow_text = None
        
        # Maximum expected lean angles for LIDAR system
        self.MAX_SIDE_LEAN = 35.0
        self.MAX_UP_DOWN_LEAN = 30.0
        
        # Maximum expected X-difference for maximum lean
        self.MAX_X_DIFF_FOR_MAX_LEAN = 4.0  # mm
        
        # LIDAR system final dart position
        self.lidar_final_tip = None
        self.last_detected_position_lidar = None
        self.frames_since_detection_lidar = 0
        
        # ================ DUAL CAMERA SYSTEM ==================
        # Static camera positions in board mm
        self.camera1_position = (0, 670)    # Front camera fixed position
        self.camera2_position = (-301, 25)  # Side camera fixed position

        # Camera settings
        self.frame_width = 640
        self.frame_height = 480

        # Camera 1 board plane line (y value in pixels where board surface is)
        self.cam1_board_plane_y = 178
        # Allowed range around the board plane for detection
        self.cam1_roi_range = 30
        # Camera 1 ROI calculated from board plane
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        # Camera 2 board plane line (y value in pixels where board surface is)
        self.cam2_board_plane_y = 200
        # Allowed range around the board plane for detection
        self.cam2_roi_range = 30
        # Camera 2 ROI calculated from board plane
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        
        # Calibration points dictionary for dual camera system
        self.calibration_points = {
            (0, 0): (310, 336),
            (-171, 0): (583, 390),
            (171, 0): (32, 294),
            (0, 171): (319, 27),
            (0, -171): (305, 571),
            (90, 50): (151, 249),
            (-20, 103): (327, 131),
            (20, -100): (277, 459),
            (90, -50): (359, 406),
        }
        
        # Add additional calibration points (from vectfix.py)
        additional_calibration_points = [
            (114, 121, 17, 153),   # Double 18
            (48, 86, 214, 182),    # Treble 18
            (119, -117, 167, 429), # Double 15
            (86, -48, 189, 359),   # Treble 15
            (-118, -121, 453, 624),# Double 7
            (-50, -88, 373, 478),  # Treble 7
            (-121, 118, 624, 240), # Double 9
            (-90, 47, 483, 42)     # Treble 9
        ]
        
        for point in additional_calibration_points:
            board_x, board_y, cam1_pixel_x, cam2_pixel_x = point
            self.calibration_points[(board_x, board_y)] = (cam1_pixel_x, cam2_pixel_x)
        
        # Create sorted mapping tables for direct interpolation
        self.cam1_pixel_to_board_mapping = []
        self.cam2_pixel_to_board_mapping = []
        
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
        
        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
        
        # Camera vectors and detected position
        self.cam1_vector = None
        self.cam2_vector = None
        self.camera_final_tip = None
        
        # Background subtractors for dual camera system
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=67, detectShadows=False)
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=40, detectShadows=False)
        
        # Detection history for smoothing in dual camera system
        self.detection_history = {
            'cam1': [],
            'cam2': [],
            'final': []
        }
        self.history_max_size = 1  # Size of history for smoothing
        
        # Camera frames for display (not shown in windows now)
        self.camera1_frame = None
        self.camera2_frame = None
        self.camera1_fg_mask = None
        self.camera2_fg_mask = None
        
        # Variables for dart persistence in dual camera system
        self.last_detected_position_camera = None
        self.frames_since_detection_camera = 0
        self.max_persistence_frames = 120  # Keep showing the dart for about 4 seconds at 30fps
        
        # =================== Combined System Methods ===================
        # Dartboard visualization settings
        self.board_scale_factor = 2.75
        self.board_extent = 171  # Updated to match provided calibration data
        self.board_radius = 170  # Standard dartboard radius in mm
        
        # Load the dartboard image
        try:
            self.dartboard_image = mpimg.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        except:
            print("Warning: dartboard image not found, using placeholder")
            self.dartboard_image = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Radii for filtering points and scoring
        self.radii = {
            "bullseye": 6.35,
            "outer_bull": 15.9,
            "inner_treble": 99,
            "outer_treble": 107,
            "inner_double": 162,
            "outer_double": 170,
            "board_edge": 195,
        }

        # Known board segments with coordinates
        self.board_segments = {
            4: (90, 50),
            5: (-20, 103),
            16: (90, -50),
            17: (20, -100),
            18: (114, 121),
            15: (119, -117),
            7: (-118, -121),
            9: (121, 118),
            1: (88, 146),
            2: (-146, -88),
            3: (-146, 88),
            6: (88, -146),
            8: (-88, -146),
            10: (0, -169),
            11: (0, 0),
            12: (-88, 146),
            13: (146, -88),
            14: (-88, 146),
            19: (-88, 146),
            20: (0, 169),
        }

        # Running flag
        self.running = True

        # Setup visualization
        self.setup_plot()

        # Initialize marker for epipolar dart placement ("CD")
        self.cd_dart, = self.ax.plot([], [], "k*", markersize=12, label="CD")

        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)

    def initialize_csv_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.csv_filename = f"dart_data_{timestamp}.csv"
        with open(self.csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                'Timestamp', 
                'LIDAR_X_mm', 'LIDAR_Y_mm', 'LIDAR_Score',
                'Camera_X_mm', 'Camera_Y_mm', 'Camera_Score',
                'Side_Lean_Angle', 'Up_Down_Lean_Angle'
            ])
        print(f"CSV logging initialized: {self.csv_filename}")

    def log_dart_data(self, lidar_position, camera_position, side_lean_angle, up_down_lean_angle):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        lidar_score = "None"
        if lidar_position:
            lidar_score = self.xy_to_dartboard_score(lidar_position[0], lidar_position[1])
        camera_score = "None"
        if camera_position:
            camera_score = self.xy_to_dartboard_score(camera_position[0], camera_position[1])
        with open(self.csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                timestamp,
                f"{lidar_position[0]:.2f}" if lidar_position else "None",
                f"{lidar_position[1]:.2f}" if lidar_position else "None",
                lidar_score,
                f"{camera_position[0]:.2f}" if camera_position else "None",
                f"{camera_position[1]:.2f}" if camera_position else "None",
                camera_score,
                f"{side_lean_angle:.2f}" if side_lean_angle is not None else "None",
                f"{up_down_lean_angle:.2f}" if up_down_lean_angle is not None else "None"
            ])

    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-400, 400)
        self.ax.set_ylim(-400, 400)
        self.ax.set_title("LIDAR and Dual Camera Dart Tracking")
        self.ax.grid(True)
        self.update_dartboard_image()
        self.ax.plot(*self.lidar1_pos, "bo", label="LIDAR 1")
        self.ax.plot(*self.lidar2_pos, "go", label="LIDAR 2")
        self.ax.plot(*self.side_camera_position, "mo", label="Side Camera")
        self.ax.plot(*self.camera1_position, "co", label="Front Camera")
        self.ax.plot(*self.camera2_position, "yo", label="Side Camera 2")
        for name, radius in self.radii.items():
            circle = plt.Circle((0, 0), radius, fill=False, linestyle='--', color='gray', alpha=0.4)
            self.ax.add_patch(circle)
            self.ax.text(0, radius, name, color='gray', fontsize=8, ha='center', va='bottom')
        self.scatter1, = self.ax.plot([], [], "b.", label="LIDAR 1 Data", zorder=3)
        self.scatter2, = self.ax.plot([], [], "g.", label="LIDAR 2 Data", zorder=3)
        self.side_camera_vector, = self.ax.plot([], [], "m--", label="Side Camera Vector")
        self.lidar1_vector, = self.ax.plot([], [], "b--", label="LIDAR 1 Vector")
        self.lidar2_vector, = self.ax.plot([], [], "g--", label="LIDAR 2 Vector")
        self.side_camera_dart, = self.ax.plot([], [], "mx", markersize=8, label="Side Camera Intersection")
        self.lidar1_dart, = self.ax.plot([], [], "bx", markersize=8, label="LIDAR 1 Projected", zorder=3)
        self.lidar2_dart, = self.ax.plot([], [], "gx", markersize=8, label="LIDAR 2 Projected", zorder=3)
        self.lidar_detected_dart, = self.ax.plot([], [], "ro", markersize=10, label="LIDAR Final Tip", zorder=10)
        self.camera1_vector, = self.ax.plot([], [], "c--", label="Front Camera Vector")
        self.camera2_vector, = self.ax.plot([], [], "y--", label="Side Camera 2 Vector")
        self.camera_detected_dart, = self.ax.plot([], [], "go", markersize=10, label="Camera Final Tip", zorder=10)
        self.lean_text = self.ax.text(-380, 380, "", fontsize=9)
        self.lidar_score_text = self.ax.text(-380, 350, "", fontsize=9, color='red')
        self.camera_score_text = self.ax.text(-380, 320, "", fontsize=9, color='green')
        self.ax.legend(loc="upper right", fontsize=8)

    def update_dartboard_image(self):
        scaled_extent = [-170 * self.board_scale_factor, 170 * self.board_scale_factor,
                         -170 * self.board_scale_factor, 170 * self.board_scale_factor]
        self.ax.imshow(self.dartboard_image, extent=scaled_extent, zorder=0)

    def xy_to_dartboard_score(self, x, y):
        scores = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        distance = np.sqrt(x * x + y * y)
        angle = np.degrees(np.arctan2(y, x))
        angle = (angle - 90 + 360) % 360
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
        elif distance <= self.radii['board_edge']:
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

    # ================== LIDAR SYSTEM METHODS ==================
    
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

    def dual_camera_detection(self):
        # Use the camera index parameters, same as in script 2
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Initialize side camera parameters (which is actually using cap2)
        self.prev_gray = None
        
        while self.running:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print("Error reading from one or both cameras.")
                time.sleep(0.1)
                continue
            
            # Process camera 1 (front camera)
            self.process_camera1_frame(frame1)
            
            # Process camera 2 (side camera)
            self.process_camera2_frame(frame2)
            
            # Process side camera functionality (using camera 2)
            self.process_side_camera_frame(frame2)
            
            # Compute camera intersection from the dual camera system
            self.camera_final_tip = self.compute_camera_intersection()
            
            if self.camera_final_tip is not None:
                smoothed_final_tip = self.apply_smoothing(self.camera_final_tip, 'final')
                if smoothed_final_tip:
                    self.camera_final_tip = smoothed_final_tip
                    self.last_detected_position_camera = smoothed_final_tip
                    self.frames_since_detection_camera = 0
            elif self.last_detected_position_camera is not None and self.frames_since_detection_camera < self.max_persistence_frames:
                self.frames_since_detection_camera += 1
        
        cap1.release()
        cap2.release()
    
    def process_camera1_frame(self, frame):
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor1.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dart_pixel_x = None
        roi_center_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        for contour in contours:
            if cv2.contourArea(contour) > 1:
                x, y, w, h = cv2.boundingRect(contour)
                dart_pixel_x = x + w // 2
                cv2.circle(roi, (dart_pixel_x, roi_center_y), 5, (0, 255, 0), -1)
                cv2.putText(roi, f"Px: {dart_pixel_x}", (dart_pixel_x + 5, roi_center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break
        if dart_pixel_x is not None:
            board_x = self.interpolate_value(dart_pixel_x, self.cam1_pixel_to_board_mapping)
            smoothed_board_x = self.apply_smoothing(board_x, 'cam1')
            self.cam1_vector = (smoothed_board_x, 0)
            cv2.putText(roi, f"Board X: {smoothed_board_x:.1f}mm", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None
        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi
        self.camera1_frame = frame_rot
        self.camera1_fg_mask = fg_mask
        return frame_rot, fg_mask

    def process_camera2_frame(self, frame):
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor2.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dart_pixel_x = None
        roi_center_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                x, y, w, h = cv2.boundingRect(contour)
                dart_pixel_x = x + w // 2
                cv2.circle(roi, (dart_pixel_x, roi_center_y), 5, (0, 255, 0), -1)
                cv2.putText(roi, f"Px: {dart_pixel_x}", (dart_pixel_x + 5, roi_center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break
        if dart_pixel_x is not None:
            board_y = self.interpolate_value(dart_pixel_x, self.cam2_pixel_to_board_mapping)
            smoothed_board_y = self.apply_smoothing(board_y, 'cam2')
            self.cam2_vector = (0, smoothed_board_y)
            cv2.putText(roi, f"Board Y: {smoothed_board_y:.1f}mm", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        self.camera2_frame = frame_rot
        self.camera2_fg_mask = fg_mask
        return frame_rot, fg_mask

    def compute_camera_intersection(self):
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
        cam1_board_x = self.cam1_vector[0]
        cam1_ray_start = self.camera1_position
        cam1_ray_end = (cam1_board_x, 0)
        cam2_board_y = self.cam2_vector[1]
        cam2_ray_start = self.camera2_position
        cam2_ray_end = (0, cam2_board_y)
        intersection = self.compute_line_intersection(
            cam1_ray_start, cam1_ray_end, 
            cam2_ray_start, cam2_ray_end
        )
        return intersection

    def process_side_camera_frame(self, frame):
        """Process side camera frame for angle detection and lean determination.
        This replaces the separate side_camera_detection method, using the same camera as camera2."""
        
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        roi = frame_rot[self.side_camera_roi_top:self.side_camera_roi_bottom, 
                       self.side_camera_roi_left:self.side_camera_roi_right]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        _, diff_thresh = cv2.threshold(frame_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray.copy()
        
        fg_mask = self.side_camera_bg_subtractor.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY)[1]
        
        combined_mask = cv2.bitwise_or(fg_mask, diff_thresh)
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        self.side_camera_data["dart_mm_y"] = None
        self.side_camera_data["dart_angle"] = None
        self.side_camera_data["tip_pixel"] = None
        
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            tip_contour = None
            tip_point = None
            for contour in contours:
                if cv2.contourArea(contour) > self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    dart_pixel_x = x + w // 2
                    roi_center_y = self.side_camera_board_plane_y - self.side_camera_roi_top
                    if tip_contour is None:
                        tip_contour = contour
                        tip_point = (dart_pixel_x, roi_center_y)
            
            if tip_contour is not None and tip_point is not None:
                dart_angle = self.measure_tip_angle(combined_mask, tip_point)
                global_pixel_x = tip_point[0] + self.side_camera_roi_left
                global_pixel_y = tip_point[1] + self.side_camera_roi_top
                dart_mm_y = -0.628 * global_pixel_x + 192.8
                self.side_camera_data["dart_mm_y"] = dart_mm_y
                self.side_camera_data["dart_angle"] = dart_angle
                self.side_camera_data["tip_pixel"] = (global_pixel_x, global_pixel_y)

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

    def detect_up_down_lean(self, lidar1_point, lidar2_point):
        if lidar1_point is not None and lidar2_point is not None:
            x1 = lidar1_point[0]
            x2 = lidar2_point[0]
            x_diff = x1 - x2
            lean_angle = (x_diff / self.MAX_X_DIFF_FOR_MAX_LEAN) * self.MAX_UP_DOWN_LEAN
            confidence = min(1.0, (len(self.lidar1_recent_points) + len(self.lidar2_recent_points)) / (2 * self.max_recent_points))
        elif (lidar1_point is not None or lidar2_point is not None) and self.side_camera_data.get("dart_angle") is not None:
            side_lean_angle = self.side_camera_data.get("dart_angle")
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
            return None
        if side_lean_angle is None:
            side_lean_angle = 90
        if up_down_lean_angle is None:
            up_down_lean_angle = 0
        original_x, original_y = lidar_point
        adjusted_y = original_y
        if camera_y is not None:
            lean_direction = 0
            if side_lean_angle < 85:
                lean_factor = (85 - side_lean_angle) / 85.0
                lean_direction = -1
            elif side_lean_angle > 95:
                lean_factor = (side_lean_angle - 95) / 85.0
                lean_direction = 1
            else:
                lean_factor = 0
            y_displacement = abs(original_y - camera_y)
            MAX_SIDE_ADJUSTMENT = 6.0
            side_adjustment = min(lean_factor * y_displacement, MAX_SIDE_ADJUSTMENT)
            adjusted_y = original_y + (side_adjustment * lean_direction)
        y_distance_from_center = abs(original_y)
        MAX_UP_DOWN_ADJUSTMENT = 4.0
        up_down_adjustment = (up_down_lean_angle / self.MAX_UP_DOWN_LEAN) * (y_distance_from_center / 170.0) * MAX_UP_DOWN_ADJUSTMENT
        adjusted_x = original_x + up_down_adjustment
        return (adjusted_x, adjusted_y)

    def find_side_camera_board_intersection(self, camera_y):
        if camera_y is None:
            return None
        return (0, camera_y)

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
            
    def calculate_lidar_final_tip_position(self, camera_point, lidar1_point, lidar2_point):
        valid_lidar_points = []
        if lidar1_point is not None:
            valid_lidar_points.append(lidar1_point)
        if lidar2_point is not None:
            valid_lidar_points.append(lidar2_point)
        if not valid_lidar_points:
            return None
        if len(valid_lidar_points) == 1:
            x, y = valid_lidar_points[0]
            if self.side_camera_data["dart_angle"] is not None:
                side_lean_angle = self.side_camera_data["dart_angle"]
                if side_lean_angle < 85:
                    lean_factor = (85 - side_lean_angle) / 85.0
                    y_correction = -lean_factor * 6.0
                    y += y_correction
                elif side_lean_angle > 95:
                    lean_factor = (side_lean_angle - 95) / 85.0
                    y_correction = lean_factor * 6.0
                    y += y_correction
            x = x * self.x_scale_correction
            y = y * self.y_scale_correction
            return (x, y)
        up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        if abs(up_down_lean_angle) > 5 and lean_confidence > 0.7:
            if up_down_lean_angle > 0:
                weight1 = 0.7
                weight2 = 0.3
            else:
                weight1 = 0.3
                weight2 = 0.7
            x = lidar1_point[0] * weight1 + lidar2_point[0] * weight2
            y = lidar1_point[1] * weight1 + lidar2_point[1] * weight2
        else:
            x = (lidar1_point[0] + lidar2_point[0]) / 2
            y = (lidar1_point[1] + lidar2_point[1]) / 2
        if self.side_camera_data["dart_angle"] is not None:
            side_lean_angle = self.side_camera_data["dart_angle"]
            if side_lean_angle < 85:
                lean_factor = (85 - side_lean_angle) / 85.0
                y_correction = -lean_factor * 6.0
                y += y_correction
            elif side_lean_angle > 95:
                lean_factor = (side_lean_angle - 95) / 85.0
                y_correction = lean_factor * 6.0
                y += y_correction
        x = x * self.x_scale_correction
        y = y * self.y_scale_correction
        return (x, y)
        
    # -------------------- Dual Camera System Methods --------------------
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
        points = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if mask[y, x] > 0:
                    points.append((x, y))
        if len(points) < min_points:
            return None
        best_angle = None
        best_inliers = 0
        for _ in range(10):
            if len(points) < 2:
                continue
            indices = np.random.choice(len(points), 2, replace=False)
            p1 = points[indices[0]]
            p2 = points[indices[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 5:
                continue
            if dx == 0:
                slope = float('inf')
                angle = 90
            else:
                slope = dy / dx
                angle_from_horizontal = math.degrees(math.atan(slope))
                angle = 90 - angle_from_horizontal
            inliers = []
            for point in points:
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
            points_arr = np.array(points)
            if len(points_arr) < 2:
                return None
            x = points_arr[:, 0]
            y = points_arr[:, 1]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator == 0:
                slope = float('inf')
                best_angle = 90
            else:
                slope = numerator / denominator
                angle_from_horizontal = math.degrees(math.atan(slope))
                best_angle = 90 - angle_from_horizontal
        return best_angle
    
    def interpolate_value(self, pixel_value, mapping_table):
        if not mapping_table:
            return None
        if pixel_value <= mapping_table[0][0]:
            return mapping_table[0][1]
        if pixel_value >= mapping_table[-1][0]:
            return mapping_table[-1][1]
        pos = bisect_left([x[0] for x in mapping_table], pixel_value)
        if pos < len(mapping_table) and mapping_table[pos][0] == pixel_value:
            return mapping_table[pos][1]
        lower_pixel, lower_value = mapping_table[pos-1]
        upper_pixel, upper_value = mapping_table[pos]
        ratio = (pixel_value - lower_pixel) / (upper_pixel - lower_pixel)
        interpolated_value = lower_value + ratio * (upper_value - lower_value)
        return interpolated_value
        
    def compute_line_intersection(self, p1, p2, p3, p4):
        denominator = ((p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0]))
        if denominator == 0:
            return None
        num_x = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[0] - p4[0])
                 - (p1[0]-p2[0]) * (p3[0]*p4[1] - p3[1]*p4[0]))
        num_y = ((p1[0]*p2[1] - p1[1]*p2[0]) * (p3[1] - p4[1])
                 - (p1[1]-p2[1]) * (p3[0]*p4[1] - p3[1]*p4[0]))
        x = num_x / denominator
        y = num_y / denominator
        return (x, y)

    def apply_smoothing(self, new_value, history_key):
        if new_value is None:
            return None
        self.detection_history[history_key].append(new_value)
        if len(self.detection_history[history_key]) > self.history_max_size:
            self.detection_history[history_key].pop(0)
        if len(self.detection_history[history_key]) >= 2:
            if history_key == 'final':
                avg_x = sum(p[0] for p in self.detection_history[history_key]) / len(self.detection_history[history_key])
                avg_y = sum(p[1] for p in self.detection_history[history_key]) / len(self.detection_history[history_key])
                return (avg_x, avg_y)
            else:
                return sum(self.detection_history[history_key]) / len(self.detection_history[history_key])
        return new_value
def update_plot(self, frame):
        lidar1_points_x = []
        lidar1_points_y = []
        lidar2_points_x = []
        lidar2_points_y = []
        while not self.lidar1_queue.empty():
            angle, distance = self.lidar1_queue.get()
            x, y = self.polar_to_cartesian(angle, distance, self.lidar1_pos, 
                                        self.lidar1_rotation, self.lidar1_mirror)
            if x is not None and y is not None:
                in_range, _ = self.filter_points_by_radii(x, y)
                if in_range:
                    lidar1_points_x.append(x)
                    lidar1_points_y.append(y)
                    self.lidar1_recent_points.append((x, y))
        while not self.lidar2_queue.empty():
            angle, distance = self.lidar2_queue.get()
            x, y = self.polar_to_cartesian(angle, distance, self.lidar2_pos, 
                                        self.lidar2_rotation, self.lidar2_mirror)
            if x is not None and y is not None:
                in_range, _ = self.filter_points_by_radii(x, y)
                if in_range:
                    lidar2_points_x.append(x)
                    lidar2_points_y.append(y)
                    self.lidar2_recent_points.append((x, y))
        self.lidar1_recent_points = self.lidar1_recent_points[-self.max_recent_points:]
        self.lidar2_recent_points = self.lidar2_recent_points[-self.max_recent_points:]
        camera_y = self.side_camera_data["dart_mm_y"]
        side_lean_angle = self.side_camera_data["dart_angle"]
        up_down_lean_angle = 0
        lean_confidence = 0
        if len(self.lidar1_recent_points) > 0 and len(self.lidar2_recent_points) > 0:
            lidar1_point = self.lidar1_recent_points[-1]
            lidar2_point = self.lidar2_recent_points[-1]
            up_down_lean_angle, lean_confidence = self.detect_up_down_lean(lidar1_point, lidar2_point)
        self.update_lean_visualization(side_lean_angle, up_down_lean_angle, lean_confidence)
        side_camera_point = self.find_side_camera_board_intersection(camera_y)
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
        lidar_final_tip = self.calculate_lidar_final_tip_position(
            side_camera_point, lidar1_projected, lidar2_projected
        )
        if lidar_final_tip is not None:
            self.lidar_final_tip = lidar_final_tip
            self.last_detected_position_lidar = lidar_final_tip
            self.frames_since_detection_lidar = 0
        elif self.last_detected_position_lidar is not None and self.frames_since_detection_lidar < self.max_persistence_frames:
            self.frames_since_detection_lidar += 1
            self.lidar_final_tip = self.last_detected_position_lidar
        else:
            self.lidar_final_tip = None
        self.log_dart_data(
            self.lidar_final_tip,
            self.camera_final_tip,
            side_lean_angle,
            up_down_lean_angle
        )
        self.scatter1.set_data(lidar1_points_x, lidar1_points_y)
        self.scatter2.set_data(lidar2_points_x, lidar2_points_y)
        if side_camera_point is not None:
            board_x = 0
            dir_x = board_x - self.side_camera_position[0]
            dir_y = side_camera_point[1] - self.side_camera_position[1]
            vector_length = np.sqrt(dir_x**2 + dir_y**2)
            if vector_length > 0:
                norm_dir_x = dir_x / vector_length
                norm_dir_y = dir_y / vector_length
            else:
                norm_dir_x, norm_dir_y = 1, 0
            extended_x = self.side_camera_position[0] + norm_dir_x * self.side_camera_vector_length
            extended_y = self.side_camera_position[1] + norm_dir_y * self.side_camera_vector_length
            self.side_camera_vector.set_data(
                [self.side_camera_position[0], extended_x],
                [self.side_camera_position[1], extended_y]
            )
            self.side_camera_dart.set_data([side_camera_point[0]], [side_camera_point[1]])
        else:
            self.side_camera_vector.set_data([], [])
            self.side_camera_dart.set_data([], [])
        if lidar1_projected is not None:
            self.lidar1_dart.set_data([lidar1_projected[0]], [lidar1_projected[1]])
        else:
            self.lidar1_dart.set_data([], [])
        if lidar2_projected is not None:
            self.lidar2_dart.set_data([lidar2_projected[0]], [lidar2_projected[1]])
        else:
            self.lidar2_dart.set_data([], [])
        if self.lidar_final_tip is not None:
            self.lidar_detected_dart.set_data([self.lidar_final_tip[0]], [self.lidar_final_tip[1]])
            lidar_score = self.xy_to_dartboard_score(self.lidar_final_tip[0], self.lidar_final_tip[1])
            lidar_description = self.get_score_description(lidar_score)
            self.lidar_score_text.set_text(f"LIDAR: {lidar_description}")
        else:
            self.lidar_detected_dart.set_data([], [])
            self.lidar_score_text.set_text("")
        if self.cam1_vector is not None:
            self.camera1_vector.set_data(
                [self.camera1_position[0], self.cam1_vector[0]], 
                [self.camera1_position[1], 0]
            )
        else:
            self.camera1_vector.set_data([], [])
        if self.cam2_vector is not None:
            self.camera2_vector.set_data(
                [self.camera2_position[0], 0], 
                [self.camera2_position[1], self.cam2_vector[1]]
            )
        else:
            self.camera2_vector.set_data([], [])
        if self.camera_final_tip is not None:
            self.camera_detected_dart.set_data([self.camera_final_tip[0]], [self.camera_final_tip[1]])
            camera_score = self.xy_to_dartboard_score(self.camera_final_tip[0], self.camera_final_tip[1])
            camera_description = self.get_score_description(camera_score)
            self.camera_score_text.set_text(f"Camera: {camera_description}")
        else:
            self.camera_detected_dart.set_data([], [])
            self.camera_score_text.set_text("")
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
        
        # Compute and update the extra epipolar dart placement ("CD")
        cd_point = self.compute_camera_intersection()  # Same computation as before (unsmoothed)
        if cd_point is not None:
            self.cd_dart.set_data([cd_point[0]], [cd_point[1]])
        else:
            self.cd_dart.set_data([], [])
        
        artists = [
            self.scatter1, self.scatter2, 
            self.side_camera_vector, self.side_camera_dart,
            self.lidar1_dart, self.lidar2_dart, 
            self.lidar_detected_dart, self.lean_text,
            self.camera1_vector, self.camera2_vector,
            self.camera_detected_dart,
            self.lidar_score_text, self.camera_score_text,
            self.cd_dart  # Added marker for CD
        ]
        
        if hasattr(self, 'lean_arrow') and self.lean_arrow:
            artists.append(self.lean_arrow)
        if hasattr(self, 'arrow_text') and self.arrow_text:
            artists.append(self.arrow_text)
            
        return artists

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
        
        # Replace the separate side_camera_thread with a single dual_camera_thread
        dual_camera_thread = threading.Thread(target=self.dual_camera_detection, daemon=True)
        
        lidar1_thread.start()
        time.sleep(1)
        lidar2_thread.start()
        time.sleep(1)
        dual_camera_thread.start()
        
        self.ani = FuncAnimation(
            self.fig, self.update_plot, 
            blit=True, interval=100, 
            cache_frame_data=False
        )
        plt.show()

    # CSV logging initialization called in update_plot after plotting
    def initialize_extra(self):
        self.x_scale_correction = 1.0
        self.y_scale_correction = 1.0
        self.initialize_csv_logging()

# Modified main function to match script 2's interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LIDAR and Dual Camera Dart Tracking')
    parser.add_argument('--cam1', type=int, default=0, help='Front camera index (default: 0)')
    parser.add_argument('--cam2', type=int, default=2, help='Side camera index (default: 2)')
    parser.add_argument('--lidar1', type=str, default="./tri_test_lidar1", help='LIDAR 1 script path')
    parser.add_argument('--lidar2', type=str, default="./tri_test_lidar2", help='LIDAR 2 script path')
    
    args = parser.parse_args()
    
    visualizer = LidarDualCameraVisualizer(
        cam_index1=args.cam1,
        cam_index2=args.cam2
    )
    visualizer.initialize_extra()
    visualizer.run(args.lidar1, args.lidar2)
