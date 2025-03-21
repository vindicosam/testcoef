import cv2
import numpy as np
import math

class DualCameraEpipolarTrainer:
    def __init__(self, cam_index1=0, cam_index2=2):
        # Static camera positions in board mm
        self.camera1_position = (0, 350)    # Front camera fixed position
        self.camera2_position = (-650, 0)   # Side camera fixed position

        # Camera settings
        self.frame_width = 640
        self.frame_height = 480

        # Camera 1 board plane line (y value in pixels where board surface is)
        self.cam1_board_plane_y = 199
        # Allowed range around the board plane for detection
        self.cam1_roi_range = 20
        # Camera 1 ROI calculated from board plane
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
        # Camera 2 board plane line (y value in pixels where board surface is)
        self.cam2_board_plane_y = 195
        # Allowed range around the board plane for detection
        self.cam2_roi_range = 20
        # Camera 2 ROI calculated from board plane
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range

        # Calibration derived from provided data points
        # For cam1: (0,0) -> x290, (-171,0) -> x506, (171,0) -> x68
        self.camera1_pixel_to_mm_factor = -0.782  # Slope in mm/pixel
        self.camera1_pixel_offset = 226.8         # Board x when pixel_x = 0
        
        # For cam2: (0,0) -> x307, (0,-171) -> x578, (0,171) -> x34
        self.camera2_pixel_to_mm_factor = -0.628  # Slope in mm/pixel
        self.camera2_pixel_offset = 192.8         # Board y when pixel_x = 0

        # Background subtractors
        self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

        # Camera indices
        self.cam_index1 = cam_index1
        self.cam_index2 = cam_index2

        # Board image and extent (in mm)
        self.board_extent = 171  # Updated to match provided calibration data
        self.board_radius = 170  # Standard dartboard radius in mm
        
        # Load the dartboard image
        self.board_image = cv2.imread("winmau-blade-6-triple-core-carbon-professional-bristle-dartboard.jpg")
        if self.board_image is None:
            self.board_image = np.zeros((500, 500, 3), dtype=np.uint8)

        # Vectors from each camera (in board mm)
        self.cam1_vector = None  # Vector from cam1 through detected point on board
        self.cam2_vector = None  # Vector from cam2 through detected point on board
        self.final_tip = None    # Intersection of the vectors

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

    def process_camera1_frame(self, frame):
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        roi = frame_rot[self.cam1_roi_top:self.cam1_roi_bottom, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor1.apply(gray)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dart_pixel_x = None
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                dart_pixel_x = x + w // 2
                # Highlight the detected point
                roi_center_y = self.cam1_board_plane_y - self.cam1_roi_top
                cv2.circle(roi, (dart_pixel_x, roi_center_y), 5, (0, 255, 0), -1)
                cv2.putText(roi, f"Px: {dart_pixel_x}", (dart_pixel_x + 5, roi_center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break

        if dart_pixel_x is not None:
            # Apply calibration for cam1 to get board x-coordinate
            board_x = self.camera1_pixel_to_mm_factor * dart_pixel_x + self.camera1_pixel_offset
            # Store vector info - we know this passes through (board_x, 0) on the board
            self.cam1_vector = (board_x, 0)
            # Add calibration debugging info to display
            cv2.putText(roi, f"Board X: {board_x:.1f}mm", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam1_vector = None

        # Draw a line at the board plane y-value
        cv2.line(roi, (0, self.cam1_board_plane_y - self.cam1_roi_top), 
                     (roi.shape[1], self.cam1_board_plane_y - self.cam1_roi_top), 
                     (0, 255, 255), 1)

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
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                dart_pixel_x = x + w // 2
                # Highlight the detected point
                roi_center_y = self.cam2_board_plane_y - self.cam2_roi_top
                cv2.circle(roi, (dart_pixel_x, roi_center_y), 5, (0, 255, 0), -1)
                cv2.putText(roi, f"Px: {dart_pixel_x}", (dart_pixel_x + 5, roi_center_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                break

        if dart_pixel_x is not None:
            # Apply calibration for cam2 to get board y-coordinate
            board_y = self.camera2_pixel_to_mm_factor * dart_pixel_x + self.camera2_pixel_offset
            # Store vector info - we know this passes through (0, board_y) on the board
            self.cam2_vector = (0, board_y)
            # Add calibration debugging info to display
            cv2.putText(roi, f"Board Y: {board_y:.1f}mm", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            self.cam2_vector = None
            
        # Draw a line at the board plane y-value
        cv2.line(roi, (0, self.cam2_board_plane_y - self.cam2_roi_top), 
                     (roi.shape[1], self.cam2_board_plane_y - self.cam2_roi_top), 
                     (0, 255, 255), 1)

        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        return frame_rot, fg_mask

    def mm_to_pixel(self, x, y, board_img):
        """Convert board mm coordinates to pixel coordinates in the board image"""
        h, w, _ = board_img.shape
        
        # Calculate center of the image
        center_x = w // 2
        center_y = h // 2
        
        # Convert board mm to image pixels (centered at the middle of the image)
        # Scale factor of 1.0 means 1mm = 1 pixel
        scale_factor = 1.0
        pixel_x = int(center_x + x * scale_factor)
        pixel_y = int(center_y - y * scale_factor)  # Y is inverted in image coordinates
        
        return pixel_x, pixel_y

    def update_board_projection(self):
        # Create a larger canvas to display the board and camera positions
        canvas_size = 1200  # Make a large canvas to fit everything
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255  # White background
        
        # Calculate the scale factor to fit the board properly
        # Dartboard radius is 170mm, we want it to take up a reasonable portion of the canvas
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
            
            # The dartboard image needs to fill the entire board circle
            # Resize board image to match the board boundary circle
            board_size = int(self.board_radius * 2 * scale_factor)
            
            # Scale factor multiplier - increase this to make the dartboard larger relative to the boundary
            image_scale_multiplier = 1.6  # Adjust this value to make the dartboard fill the boundary circle
            board_img_size = int(board_size * image_scale_multiplier)
            
            board_resized = cv2.resize(self.board_image, (board_img_size, board_img_size))
            
            # Calculate position to paste the board image (centered)
            board_x = canvas_center_x - board_img_size // 2
            board_y = canvas_center_y - board_img_size // 2
            
            # Create a circular mask for the board
            mask = np.zeros((board_img_size, board_img_size), dtype=np.uint8)
            cv2.circle(mask, (board_img_size//2, board_img_size//2), board_img_size//2, 255, -1)
            
            # Paste the board image onto the canvas
            # Make sure we don't go out of bounds
            if (board_x >= 0 and board_y >= 0 and 
                board_x + board_img_size <= canvas_size and 
                board_y + board_img_size <= canvas_size):
                canvas_roi = canvas[board_y:board_y+board_img_size, board_x:board_x+board_img_size]
                board_masked = cv2.bitwise_and(board_resized, board_resized, mask=mask)
                canvas_roi[mask > 0] = board_masked[mask > 0]
        
        # Draw reference grid for calibration
        # Draw horizontal and vertical lines at 0,0
        cv2.line(canvas, (0, canvas_center_y), (canvas_size, canvas_center_y), (200, 200, 200), 1)  # Horizontal
        cv2.line(canvas, (canvas_center_x, 0), (canvas_center_x, canvas_size), (200, 200, 200), 1)  # Vertical
        
        # Draw board boundary circle
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), int(self.board_radius * scale_factor), (0, 0, 0), 1)
        
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

        # --- Cam1 vector ---
        if self.cam1_vector is not None:
            # The point where the vector from camera 1 passes through the board
            board_point = mm_to_canvas_px(*self.cam1_vector)
            cv2.circle(canvas, board_point, 5, (0, 0, 255), -1)
            cv2.putText(canvas, f"X: {self.cam1_vector[0]:.1f}", (board_point[0]+5, board_point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw the vector from camera through the board point
            cv2.line(canvas, cam1_px, board_point, (0, 0, 255), 2)
            
            # Calculate extended vector (beyond the board point)
            dx = board_point[0] - cam1_px[0]
            dy = board_point[1] - cam1_px[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Extend vector by the same length again
                extension_factor = 2.0  # How far to extend beyond the board point
                extended_x = int(board_point[0] + dx * extension_factor)
                extended_y = int(board_point[1] + dy * extension_factor)
                extended_pt = (extended_x, extended_y)
                cv2.line(canvas, board_point, extended_pt, (0, 0, 255), 2)

        # --- Cam2 vector ---
        if self.cam2_vector is not None:
            # The point where the vector from camera 2 passes through the board
            board_point = mm_to_canvas_px(*self.cam2_vector)
            cv2.circle(canvas, board_point, 5, (255, 0, 0), -1)
            cv2.putText(canvas, f"Y: {self.cam2_vector[1]:.1f}", (board_point[0]+5, board_point[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Draw the vector from camera through the board point
            cv2.line(canvas, cam2_px, board_point, (255, 0, 0), 2)
            
            # Calculate extended vector (beyond the board point)
            dx = board_point[0] - cam2_px[0]
            dy = board_point[1] - cam2_px[1]
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Extend vector by the same length again
                extension_factor = 2.0  # How far to extend beyond the board point
                extended_x = int(board_point[0] + dx * extension_factor)
                extended_y = int(board_point[1] + dy * extension_factor)
                extended_pt = (extended_x, extended_y)
                cv2.line(canvas, board_point, extended_pt, (255, 0, 0), 2)

        # --- Final Dart Position ---
        if self.cam1_vector is not None and self.cam2_vector is not None:
            # Calculate intersection of the two vectors
            self.final_tip = self.compute_intersection()
            
            if self.final_tip is not None:
                dart_x, dart_y = self.final_tip
                
                # Check if the dart position is reasonably within the board area (with some margin)
                max_radius = self.board_radius * 1.2  # Allow detection slightly outside board
                dart_radius = math.sqrt(dart_x**2 + dart_y**2)
                
                if dart_radius <= max_radius:
                    # Add dart to history with current frame number
                    self.dart_history.append((dart_x, dart_y))
                    self.dart_lifetime[len(self.dart_history) - 1] = self.frame_counter + self.dart_persistence
                    
                    # Keep history at max size
                    if len(self.dart_history) > self.max_history:
                        # Remove oldest dart
                        self.dart_history.pop(0)
                        # Update keys in lifetime dictionary
                        new_lifetime = {}
                        for key, value in self.dart_lifetime.items():
                            if key > 0:  # Skip the one we removed
                                new_lifetime[key - 1] = value
                        self.dart_lifetime = new_lifetime
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Draw all darts in history that are still within their lifetime
        for i, (dart_x, dart_y) in enumerate(self.dart_history):
            # Skip darts that have expired
            if i not in self.dart_lifetime or self.frame_counter > self.dart_lifetime[i]:
                continue
                
            final_px = mm_to_canvas_px(dart_x, dart_y)
            
            # Calculate alpha (transparency) based on remaining lifetime
            # Fade out darts as they get closer to expiring
            if i in self.dart_lifetime:
                remaining_frames = self.dart_lifetime[i] - self.frame_counter
                alpha = min(1.0, remaining_frames / (self.dart_persistence * 0.3))
                alpha = max(0.3, alpha)  # Keep minimum visibility
            else:
                alpha = 0.5  # Default alpha for darts without lifetime
            
            # Latest dart gets highlighted
            if i == len(self.dart_history) - 1 and self.final_tip is not None:
                # Draw the intersection point
                cv2.circle(canvas, final_px, 8, (0, 0, 0), -1)  # Black outline
                cv2.circle(canvas, final_px, 6, (0, 255, 0), -1)  # Green center
                
                # Find closest segment
                closest_segment = None
                min_distance = float('inf')
                for segment, (seg_x, seg_y) in self.board_segments.items():
                    dist = math.sqrt((dart_x - seg_x)**2 + (dart_y - seg_y)**2)
                    if dist < min_distance:
                        min_distance = dist
                        closest_segment = segment
                
                # Draw text with black outline for visibility
                segment_info = f" (near Seg {closest_segment})" if closest_segment and min_distance < 50 else ""
                label = f"Dart: ({dart_x:.1f}, {dart_y:.1f}){segment_info}"
                
                cv2.putText(canvas, label, (final_px[0]+10, final_px[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(canvas, label, (final_px[0]+10, final_px[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                # Color intensity based on alpha
                intensity = int(255 * alpha)
                color = (0, intensity, 0)
                
                # Draw older darts smaller and with transparency effect
                cv2.circle(canvas, final_px, 6, (0, 0, 0), -1)  # Black outline
                cv2.circle(canvas, final_px, 4, color, -1)  # Green center with alpha
        
        return canvas
        
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

    def run(self):
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        print("Press 'q' to exit, 'c' to clear dart history.")
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print("Error reading from one or both cameras.")
                break

            proc_frame1, fg_mask1 = self.process_camera1_frame(frame1)
            proc_frame2, fg_mask2 = self.process_camera2_frame(frame2)
            board_proj = self.update_board_projection()

            cv2.imshow("Camera 1 Feed", proc_frame1)
            cv2.imshow("Camera 1 FG Mask", fg_mask1)
            cv2.imshow("Camera 2 Feed", proc_frame2)
            cv2.imshow("Camera 2 FG Mask", fg_mask2)
            cv2.imshow("Board Projection", board_proj)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Clear dart history
                print("Clearing dart history")
                self.dart_history = []
                self.dart_lifetime = {}

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    trainer = DualCameraEpipolarTrainer()
    trainer.run()
