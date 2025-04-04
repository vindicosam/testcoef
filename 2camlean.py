import cv2
import numpy as np
import math
from bisect import bisect_left
import time

class DualCameraEpipolarTrainer:
    def __init__(self, cam_index1=0, cam_index2=2):
        # Static camera positions in board mm
        self.camera1_position = (0, 650)    # Front camera fixed position
        self.camera2_position = (-425, 0)   # Side camera fixed position

        # Camera settings
        self.frame_width = 640
        self.frame_height = 480

        # Camera 1 board plane line (y value in pixels where board surface is)
        self.cam1_board_plane_y = 185
        # Allowed range around the board plane for detection
        self.cam1_roi_range = 30
        # Camera 1 ROI calculated from board plane
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        # Camera 2 board plane line (y value in pixels where board surface is)
        self.cam2_board_plane_y = 202
        # Allowed range around the board plane for detection
        self.cam2_roi_range = 30
        # Camera 2 ROI calculated from board plane
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range

        # Initialize calibration points dictionary
        # Format: (board_x, board_y): (cam1_pixel_x, cam2_pixel_x)
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

        # Additional calibration points from segments
        # Format: board_x, board_y, cam1_pixel_x, cam2_pixel_x
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
        
        # Add these to existing calibration points
        for point in additional_calibration_points:
            board_x, board_y, cam1_pixel_x, cam2_pixel_x = point
            self.calibration_points[(board_x, board_y)] = (cam1_pixel_x, cam2_pixel_x)

        # Additional segment calibration points for all doubles and triples
        # Format: board_x, board_y, cam1_pixel_x, cam2_pixel_x
        segment_calibration_points = [
            # Double segments (outer ring)
            (0, 169, 394, 31),      # Double 20 (top)
            (52, 161, 145, 80),     # Double 1
            (98, 139, 33, 133),     # Double 18
            (139, 98, 7, 189),      # Double 4
            (161, 52, 18, 241),     # Double 13
            (169, 0, 51, 296),      # Double 6 (right)
            (161, -52, 97, 349),    # Double 10
            (139, -98, 153, 405),   # Double 15
            (98, -139, 208, 462),   # Double 2
            (52, -161, 263, 517),   # Double 17
            (0, -169, 317, 567),    # Double 3 (bottom)
            (-52, -161, 371, 608),  # Double 19
            (-98, -139, 429, 629),  # Double 7
            (-139, -98, 490, 608),  # Double 16
            (-161, -52, 545, 518),  # Double 8
            (-169, 0, 592, 357),    # Double 11 (left)
            (-161, 52, 629, 209),   # Double 14
            (-139, 98, 636, 82),    # Double 9
            (-98, 139, 597, 17),    # Double 12
            (-52, 161, 486, 9),     # Double 5

            # Triple segments (middle ring)
            (0, 106, 321, 145),     # Triple 20 (top)
            (33, 101, 249, 164),    # Triple 1
            (62, 87, 191, 192),     # Triple 18
            (87, 62, 155, 227),     # Triple 4
            (101, 33, 143, 265),    # Triple 13
            (106, 0, 155, 304),     # Triple 6 (right)
            (101, -33, 175, 345),   # Triple 10
            (87, -62, 209, 382),    # Triple 15
            (62, -87, 244, 419),    # Triple 2
            (33, -101, 281, 452),   # Triple 17
            (0, -106, 321, 478),    # Triple 3 (bottom)
            (-33, -101, 360, 494),  # Triple 19
            (-62, -87, 398, 490),   # Triple 7
            (-87, -62, 436, 466),   # Triple 16
            (-101, -33, 466, 414),  # Triple 8
            (-106, 0, 489, 345),    # Triple 11 (left)
            (-101, 33, 497, 274),   # Triple 14
            (-87, 62, 490, 207),    # Triple 9
            (-62, 87, 456, 162),    # Triple 12
            (-33, 101, 395, 143),   # Triple 5
        ]
            
        # Only add segment calibration points if they have pixel values
        for point in segment_calibration_points:
            board_x, board_y, cam1_pixel_x, cam2_pixel_x = point
            if cam1_pixel_x is not None and cam2_pixel_x is not None:
                self.calibration_points[(board_x, board_y)] = (cam1_pixel_x, cam2_pixel_x)
                print(f"Added calibration point: ({board_x}, {board_y}) => Cam1: {cam1_pixel_x}, Cam2: {cam2_pixel_x}")

        # Create sorted mapping tables for direct interpolation
        # Camera 1: pixel_x to board_x mapping
        self.cam1_pixel_to_board_mapping = []
        # Camera 2: pixel_x to board_y mapping
        self.cam2_pixel_to_board_mapping = []
        
        # Fill the mapping tables
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            # We only need x mapping for camera 1
            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
            # We only need y mapping for camera 2
            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
        
        # Sort the mappings by pixel values for efficient lookup
        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
        
        # Print mapping tables for debugging
        print("\nCamera 1 pixel-to-board mapping (sorted by pixel_x):")
        for pixel_x, board_x in self.cam1_pixel_to_board_mapping:
            print(f"  Pixel: {pixel_x} -> Board X: {board_x}")
        
        # Check for pixel range coverage
        cam1_pixel_min = min(x[0] for x in self.cam1_pixel_to_board_mapping)
        cam1_pixel_max = max(x[0] for x in self.cam1_pixel_to_board_mapping)
        print(f"\nCamera 1 pixel range: {cam1_pixel_min} to {cam1_pixel_max} (width coverage: {(cam1_pixel_max - cam1_pixel_min) / self.frame_width * 100:.1f}%)")
        
        print("\nCamera 2 pixel-to-board mapping (sorted by pixel_x):")
        for pixel_x, board_y in self.cam2_pixel_to_board_mapping:
            print(f"  Pixel: {pixel_x} -> Board Y: {board_y}")
            
        # Check for pixel range coverage
        cam2_pixel_min = min(x[0] for x in self.cam2_pixel_to_board_mapping)
        cam2_pixel_max = max(x[0] for x in self.cam2_pixel_to_board_mapping)
        print(f"\nCamera 2 pixel range: {cam2_pixel_min} to {cam2_pixel_max} (width coverage: {(cam2_pixel_max - cam2_pixel_min) / self.frame_width * 100:.1f}%)")

        # Background subtractors
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=67, detectShadows=False)
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=40, detectShadows=False)

        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2

        # Board image and extent (in mm)
        self.board_extent = 171  # Updated to match provided calibration data
        self.board_radius = 170  # Standard dartboard radius in mm
        
        # Load the dartboard image
        self.board_image = cv2.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        if self.board_image is None:
            print("Warning: dartboard image not found. Using placeholder.")
            self.board_image = np.zeros((500, 500, 3), dtype=np.uint8)

        # Vectors from each camera (in board mm)
        self.cam1_vector = None  # Vector from cam1 through detected point on board
        self.cam2_vector = None  # Vector from cam2 through detected point on board
        self.final_tip = None    # Intersection of the vectors
        
        # Known board segments with coordinates
        self.board_segments = {
            4: (90, 50),
            5: (-20, 103),
            16: (90, -50),
            17: (20, -100),
            # Add new segments from provided data
            18: (114, 121),     # Double 18 
            15: (119, -117),    # Double 15
            7: (-118, -121),    # Double 7
            9: (121, 118),      # Double 9
            # Add standard segments
            1: (88, 146),       # Double 1
            2: (-146, -88),     # Double 2  
            3: (-146, 88),      # Double 3
            6: (88, -146),      # Double 6
            8: (-88, -146),     # Double 8
            10: (0, -169),      # Double 10
            11: (0, 0),         # Bullseye
            12: (-88, 146),     # Double 12
            13: (146, -88),     # Double 13
            14: (-88, 146),     # Double 14
            19: (-88, 146),     # Double 19
            20: (0, 169),       # Double 20
        }
        
        # Detection history for smoothing
        self.detection_history = {
            'cam1': [],
            'cam2': [],
            'final': []
        }
        self.history_max_size = 3  # Number of frames to keep for smoothing
        
        # Calibration mode
        self.calibration_mode = False
        self.calibration_point = None

    def interpolate_value(self, pixel_value, mapping_table):
        """
        Interpolate a value using the provided mapping table.
        mapping_table is a list of (pixel_value, board_coordinate) pairs sorted by pixel_value.
        """
        # Handle edge cases
        if not mapping_table:
            return None
        
        # If pixel value is outside the range of our mapping, use the nearest edge value
        if pixel_value <= mapping_table[0][0]:
            return mapping_table[0][1]
        if pixel_value >= mapping_table[-1][0]:
            return mapping_table[-1][1]
        
        # Find position where pixel_value would be inserted to maintain sorted order
        pos = bisect_left([x[0] for x in mapping_table], pixel_value)
        
        # If exact match
        if pos < len(mapping_table) and mapping_table[pos][0] == pixel_value:
            return mapping_table[pos][1]
        
        # Need to interpolate between pos-1 and pos
        lower_pixel, lower_value = mapping_table[pos-1]
        upper_pixel, upper_value = mapping_table[pos]
        
        # Linear interpolation
        ratio = (pixel_value - lower_pixel) / (upper_pixel - lower_pixel)
        interpolated_value = lower_value + ratio * (upper_value - lower_value)
        
        return interpolated_value

    def compute_line_intersection(self, p1, p2, p3, p4):
        """
        Compute the intersection of two lines.
        p1, p2 define the first line, p3, p4 define the second line.
        Returns the intersection point (x, y) or None if lines are parallel.
        """
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
        """
        Apply smoothing to the detection using simple moving average.
        """
        if new_value is None:
            return None
            
        # Add new value to history
        self.detection_history[history_key].append(new_value)
        
        # Keep history at max size
        if len(self.detection_history[history_key]) > self.history_max_size:
            self.detection_history[history_key].pop(0)
            
        # If we have enough history, compute average
        if len(self.detection_history[history_key]) >= 2:
            if history_key == 'final':
                # For final position, average x and y separately
                avg_x = sum(p[0] for p in self.detection_history[history_key]) / len(self.detection_history[history_key])
                avg_y = sum(p[1] for p in self.detection_history[history_key]) / len(self.detection_history[history_key])
                return (avg_x, avg_y)
            else:
                # For camera vectors, we're just smoothing a single value
                return sum(self.detection_history[history_key]) / len(self.detection_history[history_key])
        
        return new_value

    def process_camera1_frame(self, frame):
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor1.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dart_pixel_x = None
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        for contour in contours:
            if cv2.contourArea(contour) > 5:
                x, y, w, h = cv2.boundingRect(contour)
                dart_pixel_x = x + w // 2
                # Highlight the detected point
                cv2.circle(roi, (dart_pixel_x, roi_center_y), 5, (0, 255, 0), -1)
                cv2.putText(roi, f"Px: {dart_pixel_x}", (dart_pixel_x + 5, roi_center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # In calibration mode, display the pixel values for easy recording
                if self.calibration_mode and self.calibration_point:
                    print(f"Cam1 pixel for {self.calibration_point}: {dart_pixel_x}")
                
                break

        if dart_pixel_x is not None:
            # Use interpolation to get board x-coordinate directly from pixel value
            board_x = self.interpolate_value(dart_pixel_x, self.cam1_pixel_to_board_mapping)
            
            # Apply smoothing
            smoothed_board_x = self.apply_smoothing(board_x, 'cam1')
            
            # Store vector info - we know this passes through (board_x, 0) on the board
            self.cam1_vector = (smoothed_board_x, 0)
            
            # Add debugging info to display
            cv2.putText(roi, f"Board X: {smoothed_board_x:.1f}mm", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Print to terminal for monitoring
            print(f"Camera 1: Detected at pixel x={dart_pixel_x}, mapped to board X={smoothed_board_x:.1f}mm")
        else:
            self.cam1_vector = None

        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi
        return frame_rot, fg_mask

    def process_camera2_frame(self, frame):
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor2.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dart_pixel_x = None
        
        # Draw a tracking line at the board plane y-value
        roi_center_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi, (0, roi_center_y), (roi.shape[1], roi_center_y), (0, 255, 255), 1)
        
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                x, y, w, h = cv2.boundingRect(contour)
                dart_pixel_x = x + w // 2
                # Highlight the detected point
                cv2.circle(roi, (dart_pixel_x, roi_center_y), 5, (0, 255, 0), -1)
                cv2.putText(roi, f"Px: {dart_pixel_x}", (dart_pixel_x + 5, roi_center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # In calibration mode, display the pixel values for easy recording
                if self.calibration_mode and self.calibration_point:
                    print(f"Cam2 pixel for {self.calibration_point}: {dart_pixel_x}")
                
                break

        if dart_pixel_x is not None:
            # Use interpolation to get board y-coordinate directly from pixel value
            board_y = self.interpolate_value(dart_pixel_x, self.cam2_pixel_to_board_mapping)
            
            # Apply smoothing
            smoothed_board_y = self.apply_smoothing(board_y, 'cam2')
            
            # Store vector info - we know this passes through (0, board_y) on the board
            self.cam2_vector = (0, smoothed_board_y)
            
            # Add debugging info to display
            cv2.putText(roi, f"Board Y: {smoothed_board_y:.1f}mm", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Print to terminal for monitoring
            print(f"Camera 2: Detected at pixel x={dart_pixel_x}, mapped to board Y={smoothed_board_y:.1f}mm")
        else:
            self.cam2_vector = None
            
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        return frame_rot, fg_mask

    def update_board_projection(self):
        # Create a larger canvas to display the board and camera positions
        canvas_size = 1200  # Make a large canvas to fit everything
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background
        
        # Calculate the scale factor to fit the board properly
        board_px_radius = 200  # Desired radius of board in pixels on the canvas
        scale_factor = board_px_radius / self.board_radius  # px/mm conversion
        
        # Calculate center of the canvas
        canvas_center_x = canvas_size // 2
        canvas_center_y = canvas_size // 2
        
        # Function to convert from mm coordinates to canvas pixel coordinates
        def mm_to_canvas_px(x, y):
            # Origin (0,0) is at center of canvas
            px = int(canvas_center_x + x * scale_factor)
            py = int(canvas_center_y - y * scale_factor)  # Y inverted in pixel coordinates
            return (px, py)
        
        # Draw the board image centered on the canvas
        if self.board_image is not None:
            h, w = self.board_image.shape[:2]
            board_size = int(self.board_radius * 2 * scale_factor)
            image_scale_multiplier = 2.75  # Adjust to make the dartboard fill the boundary circle
            board_img_size = int(board_size * image_scale_multiplier)
            board_resized = cv2.resize(self.board_image, (board_img_size, board_img_size))
            board_x = canvas_center_x - board_img_size // 2
            board_y = canvas_center_y - board_img_size // 2
            mask = np.zeros((board_img_size, board_img_size), dtype=np.uint8)
            cv2.circle(mask, (board_img_size//2, board_img_size//2), board_img_size//2, 255, -1)
            if (board_x >= 0 and board_y >= 0 and 
                board_x + board_img_size <= canvas_size and 
                board_y + board_img_size <= canvas_size):
                canvas_roi = canvas[board_y:board_y+board_img_size, board_x:board_x+board_img_size]
                board_masked = cv2.bitwise_and(board_resized, board_resized, mask=mask)
                canvas_roi[mask > 0] = board_masked[mask > 0]
        
        # Draw reference grid for calibration
        cv2.line(canvas, (0, canvas_center_y), (canvas_size, canvas_center_y), (200, 200, 200), 1)  # Horizontal
        cv2.line(canvas, (canvas_center_x, 0), (canvas_center_x, canvas_size), (200, 200, 200), 1)  # Vertical
        
        # Draw board boundary circle
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), int(self.board_radius * scale_factor), (0, 0, 0), 1)
        
        # Draw rings for double and triple
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), int(107 * scale_factor), (0, 0, 0), 1)  # Triple ring
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), int(170 * scale_factor), (0, 0, 0), 1)  # Double ring
        
        # Draw known segment markers for reference
        for segment, (x, y) in self.board_segments.items():
            segment_px = mm_to_canvas_px(x, y)
            cv2.circle(canvas, segment_px, 5, (128, 0, 128), -1)  # Purple dot
            cv2.putText(canvas, f"Seg {segment}", (segment_px[0]+5, segment_px[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
        
        # Convert static camera positions to pixel coordinates
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        
        # Draw camera positions
        cv2.circle(canvas, cam1_px, 8, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(canvas, cam2_px, 8, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw calibration points for visualization
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            cal_px = mm_to_canvas_px(board_x, board_y)
            cv2.circle(canvas, cal_px, 3, (255, 0, 255), -1)  # Magenta dot
        
        # --- Cam1 vector ---
        if self.cam1_vector is not None:
            board_point = mm_to_canvas_px(*self.cam1_vector)
            cv2.circle(canvas, board_point, 5, (0, 0, 255), -1)
            cv2.putText(canvas, f"X: {self.cam1_vector[0]:.1f}", (board_point[0]+5, board_point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.line(canvas, cam1_px, board_point, (0, 0, 255), 2)
            dx = board_point[0] - cam1_px[0]
            dy = board_point[1] - cam1_px[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                extension_factor = 2.0
                extended_x = int(board_point[0] + dx * extension_factor)
                extended_y = int(board_point[1] + dy * extension_factor)
                extended_pt = (extended_x, extended_y)
                cv2.line(canvas, board_point, extended_pt, (0, 0, 255), 2)

        # --- Cam2 vector ---
        if self.cam2_vector is not None:
            board_point = mm_to_canvas_px(*self.cam2_vector)
            cv2.circle(canvas, board_point, 5, (255, 0, 0), -1)
            cv2.putText(canvas, f"Y: {self.cam2_vector[1]:.1f}", (board_point[0]+5, board_point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.line(canvas, cam2_px, board_point, (255, 0, 0), 2)
            dx = board_point[0] - cam2_px[0]
            dy = board_point[1] - cam2_px[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                extension_factor = 2.0
                extended_x = int(board_point[0] + dx * extension_factor)
                extended_y = int(board_point[1] + dy * extension_factor)
                extended_pt = (extended_x, extended_y)
                cv2.line(canvas, board_point, extended_pt, (255, 0, 0), 2)

        # --- Final Dart Position ---
        if self.cam1_vector is not None and self.cam2_vector is not None:
            self.final_tip = self.compute_intersection()
            if self.final_tip is not None:
                smoothed_final_tip = self.apply_smoothing(self.final_tip, 'final')
                if smoothed_final_tip:
                    dart_x, dart_y = smoothed_final_tip
                    final_px = mm_to_canvas_px(dart_x, dart_y)
                    cv2.circle(canvas, final_px, 8, (0, 0, 0), -1)
                    cv2.circle(canvas, final_px, 6, (0, 255, 0), -1)
                    
                    closest_segment = None
                    min_distance = float('inf')
                    for segment, (seg_x, seg_y) in self.board_segments.items():
                        dist = math.sqrt((dart_x - seg_x)**2 + (dart_y - seg_y)**2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_segment = segment
                    
                    distance_from_center = math.sqrt(dart_x**2 + dart_y**2)
                    in_double = 169 <= distance_from_center <= 171
                    in_treble = 105 <= distance_from_center <= 107
                    in_bullseye = distance_from_center <= 12.7
                    in_outer_bull = 12.7 < distance_from_center <= 31.8
                    hit_description = ""
                    if in_bullseye:
                        hit_description = "BULLSEYE (50)"
                    elif in_outer_bull:
                        hit_description = "OUTER BULL (25)"
                    elif in_double:
                        hit_description = f"DOUBLE {closest_segment} ({closest_segment * 2})"
                    elif in_treble:
                        hit_description = f"TREBLE {closest_segment} ({closest_segment * 3})"
                    elif closest_segment:
                        hit_description = f"SEGMENT {closest_segment}"
                    
                    segment_info = f" - {hit_description}" if hit_description else ""
                    label = f"Dart: ({dart_x:.1f}, {dart_y:.1f}){segment_info}"
                    
                    cv2.putText(canvas, label, (final_px[0]+10, final_px[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.putText(canvas, label, (final_px[0]+10, final_px[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    print(f"\nDart hit at ({dart_x:.1f}, {dart_y:.1f}) mm")
                    print(f"Distance from center: {distance_from_center:.1f} mm")
                    print(f"Hit: {hit_description}")
        else:
            self.final_tip = None

        return canvas
        
    def compute_intersection(self):
        """
        Compute the intersection of the vectors from both cameras.
        """
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

    def toggle_calibration_mode(self):
        """Toggle calibration mode on/off"""
        self.calibration_mode = not self.calibration_mode
        if self.calibration_mode:
            print("\n*** CALIBRATION MODE ACTIVATED ***")
            print("Place dart at known positions and press 'c' to capture pixel values")
            print("Press 't' to toggle calibration mode off when done")
        else:
            print("\n*** CALIBRATION MODE DEACTIVATED ***")
            self.calibration_point = None

    def set_calibration_point(self, board_x, board_y):
        """Set the current calibration point coordinates"""
        self.calibration_point = (board_x, board_y)
        print(f"\nCalibration point set to ({board_x}, {board_y})")
        print("Place dart at this position and press 'c' to capture pixel values")

    def run(self):
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        print("\n*** DUAL CAMERA DART TRACKER ***")
        print("Press 'q' to exit.")
        print("Press 't' to toggle calibration mode.")
        print("Press 'c' in calibration mode to capture current point.")
        print("Press 'r' to reset background subtractors.")
        print("Press 's' to save current calibration to file.")
        print("Press 'l' to load calibration from file.")
        
        # FPS calculation variables
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print("Error reading from one or both cameras.")
                break

            current_time = time.time()
            frame_count += 1
            if (current_time - prev_time) > 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            proc_frame1, fg_mask1 = self.process_camera1_frame(frame1)
            proc_frame2, fg_mask2 = self.process_camera2_frame(frame2)
            board_proj = self.update_board_projection()
            
            cv2.putText(board_proj, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                      
            if self.calibration_mode:
                cv2.putText(board_proj, "CALIBRATION MODE", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.calibration_point:
                    cv2.putText(board_proj, f"Current point: {self.calibration_point}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Camera 1 Feed", proc_frame1)
            cv2.imshow("Camera 1 FG Mask", fg_mask1)
            cv2.imshow("Camera 2 Feed", proc_frame2)
            cv2.imshow("Camera 2 FG Mask", fg_mask2)
            cv2.imshow("Board Projection", board_proj)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                self.toggle_calibration_mode()
            elif key == ord("c") and self.calibration_mode:
                if self.calibration_point is None:
                    print("Please set calibration point first (using number keys 0-9)")
                else:
                    cam1_pixel = None
                    cam2_pixel = None
                    if self.cam1_vector is not None:
                        for contour in cv2.findContours(fg_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                            if cv2.contourArea(contour) > 20:
                                x, y, w, h = cv2.boundingRect(contour)
                                cam1_pixel = x + w // 2
                                break
                    
                    if self.cam2_vector is not None:
                        for contour in cv2.findContours(fg_mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                            if cv2.contourArea(contour) > 20:
                                x, y, w, h = cv2.boundingRect(contour)
                                cam2_pixel = x + w // 2
                                break
                    
                    if cam1_pixel is not None and cam2_pixel is not None:
                        print(f"Calibration point: ({self.calibration_point[0]}, {self.calibration_point[1]}) => Cam1: {cam1_pixel}, Cam2: {cam2_pixel}")
                        print(f"Add to calibration points: ({self.calibration_point[0]}, {self.calibration_point[1]}, {cam1_pixel}, {cam2_pixel})")
                        self.calibration_points[self.calibration_point] = (cam1_pixel, cam2_pixel)
                        self.cam1_pixel_to_board_mapping = []
                        self.cam2_pixel_to_board_mapping = []
                        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
                            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
                            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
                        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
                        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
                    else:
                        print("Could not detect dart in one or both cameras")
            elif key == ord("r"):
                print("Resetting background subtractors")
                self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=67, detectShadows=False)
                self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=67, detectShadows=False)
            elif key == ord("s"):
                print("Saving calibration points to file")
                try:
                    import json
                    with open("dart_calibration.json", "w") as f:
                        json.dump({str(k): v for k, v in self.calibration_points.items()}, f)
                    print("Calibration saved successfully")
                except Exception as e:
                    print(f"Error saving calibration: {e}")
            elif key == ord("l"):
                print("Loading calibration points from file")
                try:
                    import json
                    import ast
                    with open("dart_calibration.json", "r") as f:
                        loaded_points = json.load(f)
                        self.calibration_points = {ast.literal_eval(k): v for k, v in loaded_points.items()}
                    self.cam1_pixel_to_board_mapping = []
                    self.cam2_pixel_to_board_mapping = []
                    for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
                        self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
                        self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
                    self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
                    self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
                    print("Calibration loaded successfully")
                except Exception as e:
                    print(f"Error loading calibration: {e}")
            elif key >= ord("0") and key <= ord("9") and self.calibration_mode:
                segment_num = key - ord("0")
                if segment_num in self.board_segments:
                    self.set_calibration_point(*self.board_segments[segment_num])
                else:
                    print(f"No segment {segment_num} defined")

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    trainer = DualCameraEpipolarTrainer()
    trainer.run()
