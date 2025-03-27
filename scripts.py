# ROI Settings for side camera (similar to camera2 in the dual setup)
self.camera_board_plane_y = 199  # The y-coordinate where the board surface is
self.camera_roi_range = 30       # How much above and below to include
self.camera_roi_top = self.camera_board_plane_y - self.camera_roi_range
self.camera_roi_bottom = self.camera_board_plane_y + self.camera_roi_range
self.pixel_to_mm_factor = -0.628  # Slope in mm/pixel (same as camera2_pixel_to_mm_factor)
self.pixel_offset = 192.8        # Board y when pixel_x = 0 (same as camera2_pixel_offset)

def camera_detection(self):
    """Detect dart tip using the camera (now positioned on the left)."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while self.running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Rotate frame 90 degrees since camera is now on the left
        # Adjust rotation direction based on your actual camera orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # NEW: Extract ROI as horizontal strip at board surface level
        roi = frame[self.camera_roi_top:self.camera_roi_bottom, :]

        # Background subtraction and thresholding
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.camera_bg_subtractor.apply(gray)
        
        # More sensitive threshold
        fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to enhance the dart
        kernel = np.ones((3,3), np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        # Reset current detection
        self.camera_data["dart_mm_y"] = None
        self.camera_data["dart_angle"] = None

        # Detect contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the rightmost point since camera is now on the left
            tip_contour = None
            tip_point = None
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Threshold for contour size
                    x, y, w, h = cv2.boundingRect(contour)
                    dart_pixel_x = x + w // 2  # Center x of contour
                    
                    # Use the board plane as the y-position
                    roi_center_y = self.camera_board_plane_y - self.camera_roi_top
                    
                    if tip_contour is None:
                        tip_contour = contour
                        tip_point = (dart_pixel_x, roi_center_y)
            
            if tip_contour is not None and tip_point is not None:
                # Calculate dart angle
                dart_angle = self.measure_tip_angle(fg_mask, tip_point)
                
                # Map pixels to mm coordinates using new conversion
                dart_mm_y = self.pixel_to_mm_factor * tip_point[0] + self.pixel_offset
                
                # Save data
                self.camera_data["dart_mm_y"] = dart_mm_y
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
