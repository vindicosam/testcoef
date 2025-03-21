import cv2
import numpy as np
import math
import sys
import signal

class DualCameraDartDetector:
    def __init__(self, cam_index1=0, cam_index2=2):
        # Camera settings
        self.frame_width = 640
        self.frame_height = 480
        
        # Static camera positions in board mm
        self.camera1_position = (0, 350)    # Front camera fixed position
        self.camera2_position = (-650, 0)   # Side camera fixed position
        
        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2
        
        # ROI settings for Camera 1
        self.cam1_board_plane_y = 198
        self.cam1_roi_range = 30
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        # ROI settings for Camera 2
        self.cam2_board_plane_y = 199
        self.cam2_roi_range = 30
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        
        # Calibration derived from provided data points
        self.camera1_pixel_to_mm_factor = -0.782  # Slope in mm/pixel
        self.camera1_pixel_offset = 226.8         # Board x when pixel_x = 0
        
        self.camera2_pixel_to_mm_factor = -0.628  # Slope in mm/pixel
        self.camera2_pixel_offset = 192.8         # Board y when pixel_x = 0
        
        # Background subtractors
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=20, detectShadows=False)
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=20, detectShadows=False)
        
        # Board settings
        self.board_radius = 170  # Standard dartboard radius in mm
        
        # Detection results
        self.cam1_vector = None  # Vector from cam1 through detected point on board
        self.cam2_vector = None  # Vector from cam2 through detected point on board
        self.final_tip = None    # Intersection of the vectors
        
        # Angle detection results
        self.cam1_angle = None
        self.cam2_angle = None
        
        # Running flag
        self.running = True
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        sys.exit(0)
    
    def detect_dart_tip(self, mask):
        """Detect the dart tip position from binary mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the highest point (lowest y-value) across all contours
        # This is the tip of the dart
        tip_point = None
        valid_contours = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 20:  # Filter out noise
                continue
                
            valid_contours.append(contour)
            for point in contour:
                x, y = point[0]
                if tip_point is None or y < tip_point[1]:
                    tip_point = (x, y)
        
        return tip_point, valid_contours
    
    def measure_tip_angle(self, mask, tip_point):
        """
        Measure the angle of the dart tip based on the tip and nearby points.
        This uses a simple approach that finds points below the tip to determine orientation.
        """
        if tip_point is None:
            return None
            
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
            return None
            
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
            
        return angle, lean, points_below
    
    def process_camera1_frame(self, frame):
        """Process the frame from Camera 1"""
        # Rotate the frame if needed
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        roi_vis = roi.copy()
        
        # Apply background subtraction
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor1.apply(gray)
        
        # Threshold and clean up the mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Detect dart tip
        tip_point, contours = self.detect_dart_tip(thresh)
        
        # Draw all contours
        if contours:
            cv2.drawContours(roi_vis, contours, -1, (0, 255, 0), 2)
        
        # Draw a line at the board plane y-value
        plane_y = self.cam1_board_plane_y - self.cam1_roi_top
        cv2.line(roi_vis, (0, plane_y), (roi.shape[1], plane_y), (0, 255, 255), 1)
        
        dart_pixel_x = None
        # Process if tip detected
        if tip_point is not None:
            # Mark the tip point
            cv2.circle(roi_vis, tip_point, 5, (0, 0, 255), -1)
            
            # Use pixel x position for dartboard mapping
            dart_pixel_x = tip_point[0]
            
            # Measure angle
            angle_info = self.measure_tip_angle(thresh, tip_point)
            
            if angle_info is not None:
                angle, lean, points_below = angle_info
                self.cam1_angle = angle
                
                # Draw all detected points
                for point in points_below:
                    cv2.circle(roi_vis, point, 1, (255, 0, 255), -1)
                
                # Draw the line fitted through these points
                if len(points_below) > 1:
                    # Use the calculated angle to draw the line
                    angle_rad = np.radians(90 - angle)  # Convert to radians from horizontal
                    slope = np.tan(angle_rad)
                    
                    # Calculate line through tip point
                    x1 = tip_point[0] - 50
                    y1 = int(tip_point[1] + slope * (x1 - tip_point[0]))
                    x2 = tip_point[0] + 50
                    y2 = int(tip_point[1] + slope * (x2 - tip_point[0]))
                    
                    # Draw line
                    cv2.line(roi_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Display angle info on ROI
                roi_text = f"Angle: {angle:.1f}° ({lean})"
                cv2.putText(roi_vis, roi_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            self.cam1_angle = None
        
        # Convert dart_pixel_x to board coordinates
        if dart_pixel_x is not None:
            # Apply calibration for cam1 to get board x-coordinate
            board_x = self.camera1_pixel_to_mm_factor * dart_pixel_x + self.camera1_pixel_offset
            # Store vector info - we know this passes through (board_x, 0) on the board
            self.cam1_vector = (board_x, 0)
            # Add calibration debugging info to display
            cv2.putText(roi_vis, f"Board X: {board_x:.1f}mm", (10, roi_vis.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None
        
        # Copy ROI visualization back to the frame
        frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :] = roi_vis
        
        return frame_rot, thresh
    
    def process_camera2_frame(self, frame):
        """Process the frame from Camera 2"""
        # Rotate the frame if needed
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :]
        roi_vis = roi.copy()
        
        # Apply background subtraction
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor2.apply(gray)
        
        # Threshold and clean up the mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Detect dart tip
        tip_point, contours = self.detect_dart_tip(thresh)
        
        # Draw all contours
        if contours:
            cv2.drawContours(roi_vis, contours, -1, (0, 255, 0), 2)
        
        # Draw a line at the board plane y-value
        plane_y = self.cam2_board_plane_y - self.cam2_roi_top
        cv2.line(roi_vis, (0, plane_y), (roi.shape[1], plane_y), (0, 255, 255), 1)
        
        dart_pixel_x = None
        # Process if tip detected
        if tip_point is not None:
            # Mark the tip point
            cv2.circle(roi_vis, tip_point, 5, (0, 0, 255), -1)
            
            # Use pixel x position for dartboard mapping
            dart_pixel_x = tip_point[0]
            
            # Measure angle
            angle_info = self.measure_tip_angle(thresh, tip_point)
            
            if angle_info is not None:
                angle, lean, points_below = angle_info
                self.cam2_angle = angle
                
                # Draw all detected points
                for point in points_below:
                    cv2.circle(roi_vis, point, 1, (255, 0, 255), -1)
                
                # Draw the line fitted through these points
                if len(points_below) > 1:
                    # Use the calculated angle to draw the line
                    angle_rad = np.radians(90 - angle)  # Convert to radians from horizontal
                    slope = np.tan(angle_rad)
                    
                    # Calculate line through tip point
                    x1 = tip_point[0] - 50
                    y1 = int(tip_point[1] + slope * (x1 - tip_point[0]))
                    x2 = tip_point[0] + 50
                    y2 = int(tip_point[1] + slope * (x2 - tip_point[0]))
                    
                    # Draw line
                    cv2.line(roi_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Display angle info on ROI
                roi_text = f"Angle: {angle:.1f}° ({lean})"
                cv2.putText(roi_vis, roi_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            self.cam2_angle = None
        
        # Convert dart_pixel_x to board coordinates
        if dart_pixel_x is not None:
            # Apply calibration for cam2 to get board y-coordinate
            board_y = self.camera2_pixel_to_mm_factor * dart_pixel_x + self.camera2_pixel_offset
            # Store vector info - we know this passes through (0, board_y) on the board
            self.cam2_vector = (0, board_y)
            # Add calibration debugging info to display
            cv2.putText(roi_vis, f"Board Y: {board_y:.1f}mm", (10, roi_vis.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
        
        # Copy ROI visualization back to the frame
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi_vis
        
        return frame_rot, thresh
    
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
    
    def compute_intersection(self):
        """
        Compute the intersection of the vectors from both cameras.
        This finds the 3D position of the dart tip.
        """
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
            
        # For cam1, we've determined the board_x value where the vector passes through the board plane
        # Create a ray from camera1 through this point
        cam1_board_x = self.cam1_vector[0]
        cam1_ray_start = self.camera1_position
        cam1_ray_end = (cam1_board_x, 0)  # This is where the ray passes through the board
        
        # For cam2, we've determined the board_y value where the vector passes through the board plane
        # Create a ray from camera2 through this point
        cam2_board_y = self.cam2_vector[1]
        cam2_ray_start = self.camera2_position
        cam2_ray_end = (0, cam2_board_y)  # This is where the ray passes through the board
        
        # Find the intersection of these rays
        intersection = self.compute_line_intersection(
            cam1_ray_start, cam1_ray_end, 
            cam2_ray_start, cam2_ray_end
        )
        
        return intersection
    
    def create_board_visualization(self):
        """Create a simple visualization of the board with dart position"""
        # Create a canvas for the dartboard visualization
        canvas_size = 600
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background
        
        # Calculate scale factor (px/mm)
        scale_factor = 1.5  # Adjust this to control size
        
        # Center of canvas
        center_x = canvas_size // 2
        center_y = canvas_size // 2
        
        # Function to convert board mm to canvas pixels
        def mm_to_canvas_px(x, y):
            px = int(center_x + x * scale_factor)
            py = int(center_y - y * scale_factor)  # Y inverted in pixel coordinates
            return (px, py)
        
        # Draw board boundary circle
        cv2.circle(canvas, (center_x, center_y), int(self.board_radius * scale_factor), (0, 0, 0), 2)
        
        # Draw coordinate axes
        cv2.line(canvas, (center_x, 0), (center_x, canvas_size), (200, 200, 200), 1)  # Y-axis
        cv2.line(canvas, (0, center_y), (canvas_size, center_y), (200, 200, 200), 1)  # X-axis
        
        # Draw camera positions
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cv2.circle(canvas, cam1_px, 8, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        cv2.circle(canvas, cam2_px, 8, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw vectors if available
        if self.cam1_vector is not None:
            board_point = mm_to_canvas_px(*self.cam1_vector)
            cv2.circle(canvas, board_point, 5, (0, 0, 255), -1)
            cv2.line(canvas, cam1_px, board_point, (0, 0, 255), 2)
            
            # Draw angle information if available
            if self.cam1_angle is not None:
                angle_text = f"Cam1 Angle: {self.cam1_angle:.1f}°"
                cv2.putText(canvas, angle_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if self.cam2_vector is not None:
            board_point = mm_to_canvas_px(*self.cam2_vector)
            cv2.circle(canvas, board_point, 5, (255, 0, 0), -1)
            cv2.line(canvas, cam2_px, board_point, (255, 0, 0), 2)
            
            # Draw angle information if available
            if self.cam2_angle is not None:
                angle_text = f"Cam2 Angle: {self.cam2_angle:.1f}°"
                cv2.putText(canvas, angle_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw intersection point if available
        if self.final_tip is not None:
            dart_px = mm_to_canvas_px(*self.final_tip)
            cv2.circle(canvas, dart_px, 8, (0, 0, 0), -1)  # Black outline
            cv2.circle(canvas, dart_px, 6, (0, 255, 0), -1)  # Green center
            
            # Add text with coordinates
            text = f"Dart: ({self.final_tip[0]:.1f}, {self.final_tip[1]:.1f})"
            cv2.putText(canvas, text, (dart_px[0]+10, dart_px[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Outline
            cv2.putText(canvas, text, (dart_px[0]+10, dart_px[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Text
        
        return canvas
    
    def run(self):
        """Main processing loop"""
        # Open cameras
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both cameras.")
            return
        
        print("Camera opened successfully. Press 'q' to exit.")
        
        try:
            while self.running:
                # Read frames
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    print("Error: Failed to grab frames.")
                    break
                
                # Process frames
                proc_frame1, mask1 = self.process_camera1_frame(frame1)
                proc_frame2, mask2 = self.process_camera2_frame(frame2)
                
                # Calculate intersection of camera vectors
                if self.cam1_vector is not None and self.cam2_vector is not None:
                    self.final_tip = self.compute_intersection()
                else:
                    self.final_tip = None
                
                # Create board visualization
                board_vis = self.create_board_visualization()
                
                # Display results
                cv2.imshow("Camera 1", proc_frame1)
                cv2.imshow("Camera 1 Mask", mask1)
                cv2.imshow("Camera 2", proc_frame2)
                cv2.imshow("Camera 2 Mask", mask2)
                cv2.imshow("Board Visualization", board_vis)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap1.release()
            cap2.release()
            cv2.destroyAllWindows()
            print("Cameras released.")

if __name__ == "__main__":
    detector = DualCameraDartDetector()
    detector.run()
