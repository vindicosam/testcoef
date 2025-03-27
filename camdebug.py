import cv2
import numpy as np
import math
import time

class CameraDebugger:
    def __init__(self):
        # Camera configuration
        self.camera_board_plane_y = 247  # The y-coordinate where the board surface is
        self.camera_roi_range = 30       # How much above and below to include
        self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
        self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
        self.camera_roi_left = 121       # Left boundary
        self.camera_roi_right = 590      # Right boundary
        
        # Default linear calibration for pixel-to-mm conversion
        self.pixel_to_mm_factor = -0.628  # Slope in mm/pixel 
        self.pixel_offset = 192.8        # Board y when pixel_x = 0
        
        # Detection persistence
        self.last_valid_detection = {"dart_pixel_x": None, "dart_angle": None, "dart_mm_y": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30
        
        # Background subtractor with adjustable parameters
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=75, varThreshold=50, detectShadows=False
        )
        
        # Flags and storage
        self.running = True
        self.camera_data = {"dart_pixel_x": None, "dart_angle": None, "dart_mm_y": None}
        
        # Visualization flags - enable/disable different components
        self.show_original = True
        self.show_roi = True
        self.show_background = True
        self.show_angle = True
        self.show_epipolar = True
        
        # Create windows
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Background Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Board Visualization", cv2.WINDOW_NORMAL)
        
        # Trackbars for parameter tuning
        cv2.createTrackbar("ROI Top", "ROI", self.camera_roi_top, 480, self.update_roi_top)
        cv2.createTrackbar("ROI Bottom", "ROI", self.camera_roi_bottom, 480, self.update_roi_bottom)
        cv2.createTrackbar("ROI Left", "ROI", self.camera_roi_left, 640, self.update_roi_left)
        cv2.createTrackbar("ROI Right", "ROI", self.camera_roi_right, 640, self.update_roi_right)
        cv2.createTrackbar("BG History", "Background Mask", 75, 200, self.update_bg_history)
        cv2.createTrackbar("BG Threshold", "Background Mask", 50, 100, self.update_bg_threshold)
        
    def update_roi_top(self, value):
        self.camera_roi_top = value
        
    def update_roi_bottom(self, value):
        self.camera_roi_bottom = value
    
    def update_roi_left(self, value):
        self.camera_roi_left = value
        
    def update_roi_right(self, value):
        self.camera_roi_right = value
        
    def update_bg_history(self, value):
        # Recreate the background subtractor with new history parameter
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=value, 
            varThreshold=self.camera_bg_subtractor.getVarThreshold(),
            detectShadows=False
        )
        
    def update_bg_threshold(self, value):
        # Recreate the background subtractor with new threshold parameter
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.camera_bg_subtractor.getHistory(),
            varThreshold=value,
            detectShadows=False
        )
    
    def pixel_to_mm(self, pixel_x):
        """
        Convert pixel x-coordinate to mm y-coordinate using linear equation.
        """
        return self.pixel_to_mm_factor * pixel_x + self.pixel_offset
        
    def measure_tip_angle(self, mask, tip_point):
        """
        Measure the angle of the dart tip using RANSAC fitting.
        
        Args:
            mask: Binary mask containing dart
            tip_point: Detected tip coordinates (x,y)
            
        Returns:
            angle: Angle in degrees (90° = vertical) or None if angle couldn't be calculated
        """
        if tip_point is None:
            return None
            
        tip_x, tip_y = tip_point
        
        # Define search parameters
        search_depth = 25  # How far to search from the tip
        search_width = 40  # Width of search area
        min_points = 8     # Minimum points needed for reliable angle
        
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
            
        # Use RANSAC for robust angle estimation
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
                # Store inlier points for visualization
                self.inlier_points = inliers
        
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
            # No inlier points for visualization in this case
            self.inlier_points = points_right
        
        return best_angle
    
    def draw_angle_line(self, roi_image, tip_point, angle):
        """Draw a line showing the dart angle on the image."""
        if tip_point is None or angle is None:
            return roi_image
            
        # Copy image to avoid modifying original
        vis_img = roi_image.copy()
        
        # Line length
        line_length = 30
        
        # Calculate line endpoint based on angle
        # angle of 90° means vertical (pointing right)
        radians = math.radians(90 - angle)  # Convert to radians from vertical
        end_x = int(tip_point[0] + line_length * math.cos(radians))
        end_y = int(tip_point[1] + line_length * math.sin(radians))
        
        # Draw the line
        cv2.line(vis_img, tip_point, (end_x, end_y), (0, 0, 255), 2)
        
        # Add angle text
        cv2.putText(vis_img, f"{angle:.1f}°", (tip_point[0] + 5, tip_point[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw inlier points if available
        if hasattr(self, 'inlier_points') and self.inlier_points:
            for point in self.inlier_points:
                cv2.circle(vis_img, point, 1, (0, 255, 0), -1)
        
        return vis_img
    
    def draw_epipolar_line(self, width=400, height=400, dart_y=None):
        """Create a visualization of the dartboard with the epipolar line."""
        # Create a blank image to represent the dartboard
        board_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw board center
        center = (width // 2, height // 2)
        cv2.circle(board_img, center, 5, (0, 0, 255), -1)
        
        # Draw concentric circles to represent dartboard rings
        radii = [20, 45, 100, 170, 200]  # sample radii
        for radius in radii:
            cv2.circle(board_img, center, radius, (200, 200, 200), 1)
        
        # Draw epipolar line (horizontal line through y=dart_y)
        if dart_y is not None:
            # Scale from mm to pixels for visualization
            # Assuming the dartboard is roughly 400x400mm and centered
            scale_factor = height / 400
            pixel_y = int(height // 2 - dart_y * scale_factor)
            
            if 0 <= pixel_y < height:
                cv2.line(board_img, (0, pixel_y), (width-1, pixel_y), (0, 0, 255), 2)
                cv2.putText(board_img, f"y={dart_y:.1f}mm", (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return board_img
    
    def run(self):
        """Main camera detection loop with visualizations."""
        # Initialize the camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Wait a bit for camera to initialize
        time.sleep(1)
        
        print("Camera initialized. Press 'q' to quit.")
        print("Press 's' to take a snapshot of the current frame.")
        print("Press 'r' to reset the background model.")
        
        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from camera.")
                break
            
            # Rotate frame 90 degrees - adjust as needed for your camera orientation
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Show original frame with ROI rectangle marked
            if self.show_original:
                original_vis = frame.copy()
                cv2.rectangle(original_vis, 
                              (self.camera_roi_left, self.camera_roi_top), 
                              (self.camera_roi_right, self.camera_roi_bottom), 
                              (0, 255, 0), 2)
                cv2.imshow("Camera Feed", original_vis)
            
            # Extract ROI
            roi = frame[self.camera_roi_top:self.camera_roi_bottom, 
                        self.camera_roi_left:self.camera_roi_right]
            
            # Background subtraction
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            fg_mask = self.camera_bg_subtractor.apply(gray)
            
            # Thresholding
            fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)[1]
            
            # Morphological operations to enhance the dart
            kernel = np.ones((3,3), np.uint8)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
            
            # Reset current detection
            self.camera_data["dart_pixel_x"] = None
            self.camera_data["dart_angle"] = None
            self.camera_data["dart_mm_y"] = None
            
            # Create a copy of the ROI for visualization
            roi_vis = roi.copy()
            
            # Detect contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find contour with suitable size
                tip_contour = None
                tip_point = None
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Threshold for contour size
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Draw contour and bounding box
                        cv2.drawContours(roi_vis, [contour], -1, (0, 255, 0), 1)
                        cv2.rectangle(roi_vis, (x, y), (x+w, y+h), (255, 0, 0), 1)
                        
                        # Use center as dart tip
                        dart_pixel_x = x + w // 2
                        
                        # Use the board plane as the y-position
                        roi_center_y = self.camera_board_plane_y - self.camera_roi_top
                        
                        if tip_contour is None:
                            tip_contour = contour
                            tip_point = (dart_pixel_x, roi_center_y)
                
                if tip_contour is not None and tip_point is not None:
                    # Mark the tip point
                    cv2.circle(roi_vis, tip_point, 4, (0, 0, 255), -1)
                    
                    # Calculate dart angle
                    dart_angle = self.measure_tip_angle(fg_mask, tip_point)
                    
                    # Draw the angle line if angle was successfully calculated
                    if dart_angle is not None:
                        roi_vis = self.draw_angle_line(roi_vis, tip_point, dart_angle)
                    
                    # Convert to global pixel coordinates
                    global_pixel_x = tip_point[0] + self.camera_roi_left
                    
                    # Map pixels to mm coordinates
                    dart_mm_y = self.pixel_to_mm(global_pixel_x)
                    
                    # Save data
                    self.camera_data["dart_pixel_x"] = global_pixel_x
                    self.camera_data["dart_angle"] = dart_angle
                    self.camera_data["dart_mm_y"] = dart_mm_y
                    
                    # Update persistence
                    self.last_valid_detection = self.camera_data.copy()
                    self.detection_persistence_counter = self.detection_persistence_frames
                    
                    # Print detection data
                    print(f"Frame {frame_count}: Detected dart - pixel_x: {global_pixel_x}, " + 
                          f"angle: {dart_angle:.1f}°, mm_y: {dart_mm_y:.1f}mm")
            
            # If no dart detected but we have a valid previous detection
            elif self.detection_persistence_counter > 0:
                self.detection_persistence_counter -= 1
                if self.detection_persistence_counter > 0:
                    self.camera_data = self.last_valid_detection.copy()
                    print(f"Frame {frame_count}: Using persistence - " + 
                          f"pixel_x: {self.camera_data['dart_pixel_x']}, " +
                          f"angle: {self.camera_data['dart_angle']:.1f}°, " +
                          f"mm_y: {self.camera_data['dart_mm_y']:.1f}mm")
            
            # Show ROI with detection visualization
            if self.show_roi:
                cv2.imshow("ROI", roi_vis)
            
            # Show background mask
            if self.show_background:
                # Convert to color to make it more visible
                fg_mask_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Background Mask", fg_mask_vis)
            
            # Show epipolar line visualization 
            if self.show_epipolar:
                dart_mm_y = self.camera_data.get("dart_mm_y")
                board_vis = self.draw_epipolar_line(dart_y=dart_mm_y)
                cv2.imshow("Board Visualization", board_vis)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                # Save snapshot of all current views
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"camera_feed_{timestamp}.png", original_vis)
                cv2.imwrite(f"roi_view_{timestamp}.png", roi_vis)
                cv2.imwrite(f"bg_mask_{timestamp}.png", fg_mask)
                if self.show_epipolar and dart_mm_y is not None:
                    cv2.imwrite(f"board_vis_{timestamp}.png", board_vis)
                print(f"Saved snapshots with timestamp {timestamp}")
            elif key == ord('r'):
                # Reset background model
                self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=self.camera_bg_subtractor.getHistory(),
                    varThreshold=self.camera_bg_subtractor.getVarThreshold(),
                    detectShadows=False
                )
                print("Background model reset")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera detection stopped.")

if __name__ == "__main__":
    debugger = CameraDebugger()
    debugger.run()
