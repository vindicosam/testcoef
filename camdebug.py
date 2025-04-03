# Copy ROI back to rotated frame
        frame_rot[self.cam2_roi_top:self.cam2_roi_bottom, :] = roi
        
        return frame_rot, fg_mask, dart_pixel_x
    
    def compute_intersection(self):
        """
        Compute intersection of the vectors from both cameras.
        Enhanced version with additional validation.
        """
        if self.cam1_vector is None or self.cam2_vector is None:
            return None
            
        # For cam1, we've determined the board_x value
        cam1_board_x = self.cam1_vector[0]
        cam1_ray_start = self.camera1_position
        cam1_ray_end = (cam1_board_x, 0)
        
        # For cam2, we've determined the board_y value
        cam2_board_y = self.cam2_vector[1]
        cam2_ray_start = self.camera2_position
        cam2_ray_end = (0, cam2_board_y)
        
        # Find the intersection of these rays
        intersection = self.compute_line_intersection(
            cam1_ray_start, cam1_ray_end,
            cam2_ray_start, cam2_ray_end
        )
        
        # Validate intersection is within reasonable bounds of the board
        if intersection:
            x, y = intersection
            distance_from_center = math.sqrt(x*x + y*y)
            
            # Check if the point is within a reasonable distance from the board
            if distance_from_center <= self.board_radius + 10:  # 10mm margin
                return intersection
        
        return None
    
    def compute_line_intersection(self, p1, p2, p3, p4):
        """
        Compute the intersection of two lines.
        Enhanced version for better numerical stability.
        """
        # Convert inputs to numpy arrays for better handling
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        p3 = np.array(p3, dtype=np.float64)
        p4 = np.array(p4, dtype=np.float64)
        
        # Calculate direction vectors
        v1 = p2 - p1
        v2 = p4 - p3
        
        # Calculate cross product to check if lines are parallel
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        
        # If lines are parallel (or nearly parallel)
        if abs(cross_product) < 1e-10:
            return None
        
        # Calculate intersection parameter for first line
        v3 = p1 - p3
        t = (v2[0] * v3[1] - v2[1] * v3[0]) / cross_product
        
        # Calculate intersection point
        intersection = p1 + t * v1
        
        return tuple(intersection)
    
    def calculate_score(self, position):
        """
        Calculate score based on dart position
        
        Args:
            position: (x, y) position in board coordinates
            
        Returns:
            score: Score value
            description: Text description of the hit
        """
        if position is None:
            return 0, "No hit"
            
        x, y = position
        distance_from_center = math.sqrt(x*x + y*y)
        
        # Check if dart is outside the board
        if distance_from_center > self.board_radius:
            return 0, "Outside board"
            
        # Check bullseye
        if distance_from_center <= 12.7:  # Inner bullseye
            return 50, "Bullseye (50)"
        elif distance_from_center <= 31.8:  # Outer bullseye
            return 25, "Outer bull (25)"
            
        # Determine segment number based on angle
        angle_rad = math.atan2(y, x)
        angle_deg = math.degrees(angle_rad)
        
        # Convert to 0-360 range
        if angle_deg < 0:
            angle_deg += 360
            
        # Calculate segment index (20 segments, starting from top and going clockwise)
        segment_angle = (450 - angle_deg) % 360
        segment_index = int(segment_angle / 18)  # Each segment is 18 degrees
        
        # Map to dartboard segment numbers
        segment_map = [
            20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5
        ]
        
        segment_number = segment_map[segment_index % 20]
        
        # Determine multiplier based on distance
        if 169 <= distance_from_center <= 171:  # Double ring
            multiplier = 2
            hit_type = "Double"
        elif 105 <= distance_from_center <= 107:  # Triple ring
            multiplier = 3
            hit_type = "Triple"
        else:  # Single
            multiplier = 1
            hit_type = "Single"
            
        score = segment_number * multiplier
        
        if multiplier > 1:
            description = f"{hit_type} {segment_number} ({score})"
        else:
            description = f"{segment_number}"
            
        return score, description
    
    def update_board_projection(self, cam1_pixel_x=None, cam2_pixel_x=None):
        """
        Update board projection with current dart position
        Enhanced version with improved visualization and accuracy
        
        Args:
            cam1_pixel_x, cam2_pixel_x: Raw pixel values from cameras (optional)
        """
        # Create a larger canvas to display the board and camera positions
        canvas_size = 1200
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Calculate the scale factor to fit the board properly
        board_px_radius = 200
        scale_factor = board_px_radius / self.board_radius
        
        # Calculate center of the canvas
        canvas_center_x = canvas_size // 2
        canvas_center_y = canvas_size // 2
        
        # Function to convert from mm coordinates to canvas pixel coordinates
        def mm_to_canvas_px(x, y):
            px = int(canvas_center_x + x * scale_factor)
            py = int(canvas_center_y - y * scale_factor)
            return (px, py)
        
        # Draw the board image
        if self.board_image is not None:
            # Calculate size to make the dartboard fill the boundary circle
            board_size = int(self.board_radius * 2 * scale_factor)
            image_scale_multiplier = 2.75  # Adjust to make dartboard fill boundary
            board_img_size = int(board_size * image_scale_multiplier)
            
            board_resized = cv2.resize(self.board_image, (board_img_size, board_img_size))
            
            # Calculate position to paste the board image (centered)
            board_x = canvas_center_x - board_img_size // 2
            board_y = canvas_center_y - board_img_size // 2
            
            # Create circular mask
            mask = np.zeros((board_img_size, board_img_size), dtype=np.uint8)
            cv2.circle(mask, (board_img_size//2, board_img_size//2), board_img_size//2, 255, -1)
            
            # Paste board image with mask
            if (board_x >= 0 and board_y >= 0 and 
                board_x + board_img_size <= canvas_size and 
                board_y + board_img_size <= canvas_size):
                canvas_roi = canvas[board_y:board_y+board_img_size, board_x:board_x+board_img_size]
                board_masked = cv2.bitwise_and(board_resized, board_resized, mask=mask)
                canvas_roi[mask > 0] = board_masked[mask > 0]
        
        # Draw reference grid and board circles
        cv2.line(canvas, (0, canvas_center_y), (canvas_size, canvas_center_y), (200, 200, 200), 1)
        cv2.line(canvas, (canvas_center_x, 0), (canvas_center_x, canvas_size), (200, 200, 200), 1)
        
        # Draw board boundary and rings
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(self.board_radius * scale_factor), (0, 0, 0), 1)
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(107 * scale_factor), (0, 0, 0), 1)  # Triple ring
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(170 * scale_factor), (0, 0, 0), 1)  # Double ring
        
        # Draw bullseye rings
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(12.7 * scale_factor), (0, 0, 0), 1)  # Inner bull
        cv2.circle(canvas, (canvas_center_x, canvas_center_y), 
                 int(31.8 * scale_factor), (0, 0, 0), 1)  # Outer bull
        
        # Draw segment markers
        for segment, (x, y) in self.board_segments.items():
            segment_px = mm_to_canvas_px(x, y)
            cv2.circle(canvas, segment_px, 3, (128, 0, 128), -1)
            cv2.putText(canvas, f"{segment}", (segment_px[0]+5, segment_px[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
        
        # Draw camera positions
        cam1_px = mm_to_canvas_px(*self.camera1_position)
        cam2_px = mm_to_canvas_px(*self.camera2_position)
        
        cv2.circle(canvas, cam1_px, 8, (0, 255, 255), -1)
        cv2.putText(canvas, "Cam1", (cam1_px[0]+10, cam1_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(canvas, cam2_px, 8, (255, 255, 0), -1)
        cv2.putText(canvas, "Cam2", (cam2_px[0]+10, cam2_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Advanced interpolation when both cameras have detections
        if cam1_pixel_x is not None and cam2_pixel_x is not None and self.use_advanced_interpolation:
            # Perform advanced interpolation
            board_x, board_y = self.advanced_interpolation(cam1_pixel_x, cam2_pixel_x)
            
            # Apply Kalman filtering for smooth tracking
            if board_x is not None and board_y is not None:
                # Update Kalman filter
                filtered_position = self.update_kalman((board_x, board_y))
                
                if filtered_position is not None:
                    # Apply additional smoothing if needed
                    smoothed_position = self.apply_smoothing(filtered_position, 'final')
                    
                    if smoothed_position is not None:
                        # Store final position
                        self.final_tip = smoothed_position
                        
                        # Check for stable position
                        if self.last_stable_position is not None:
                            # Calculate movement from last stable position
                            dx = smoothed_position[0] - self.last_stable_position[0]
                            dy = smoothed_position[1] - self.last_stable_position[1]
                            movement = math.sqrt(dx*dx + dy*dy)
                            
                            # If movement is below threshold, increment stable counter
                            if movement < self.movement_threshold:
                                self.stable_position_count += 1
                            else:
                                self.stable_position_count = 0
                        
                        # Update last stable position
                        self.last_stable_position = smoothed_position
                        
                        # Draw the dart position on board
                        dart_px = mm_to_canvas_px(*smoothed_position)
                        
                        # Color based on stability
                        if self.stable_position_count >= 10:
                            # Stable position (yellow)
                            color = (0, 255, 255)
                        else:
                            # Moving position (green)
                            color = (0, 255, 0)
                        
                        # Draw dart position
                        cv2.circle(canvas, dart_px, 8, (0, 0, 0), -1)  # Black outline
                        cv2.circle(canvas, dart_px, 6, color, -1)  # Colored center
                        
                        # Display coordinates
                        x, y = smoothed_position
                        cv2.putText(canvas, f"({x:.1f}, {y:.1f})", (dart_px[0]+10, dart_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(canvas, f"({x:.1f}, {y:.1f})", (dart_px[0]+10, dart_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # Calculate and display score
                        score, description = self.calculate_score(smoothed_position)
                        
                        score_text = f"Score: {score} - {description}"
                        cv2.putText(canvas, score_text, (dart_px[0]+10, dart_px[1]+25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(canvas, score_text, (dart_px[0]+10, dart_px[1]+25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        
                        # If position is stable for a while, display "STABLE" indicator
                        if self.stable_position_count >= 10:
                            cv2.putText(canvas, "STABLE", (dart_px[0]-20, dart_px[1]-20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            # Standard processing from original code
            # --- Cam1 vector ---
            if self.cam1_vector is not None:
                board_point = mm_to_canvas_px(*self.cam1_vector)
                cv2.circle(canvas, board_point, 5, (0, 0, 255), -1)
                cv2.putText(canvas, f"X: {self.cam1_vector[0]:.1f}", (board_point[0]+5, board_point[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Draw vector from camera
                cv2.line(canvas, cam1_px, board_point, (0, 0, 255), 2)
                
                # Calculate and draw extended vector
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
                
                # Draw vector from camera
                cv2.line(canvas, cam2_px, board_point, (255, 0, 0), 2)
                
                # Calculate and draw extended vector
                dx = board_point[0] - cam2_px[0]
                dy = board_point[1] - cam2_px[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    extension_factor = 2.0
                    extended_x = int(board_point[0] + dx * extension_factor)
                    extended_y = int(board_point[1] + dy * extension_factor)
                    extended_pt = (extended_x, extended_y)
                    cv2.line(canvas, board_point, extended_pt, (255, 0, 0), 2)
            
            # --- Final Dart Position from vector intersection ---
            if self.cam1_vector is not None and self.cam2_vector is not None:
                # Calculate intersection
                self.final_tip = self.compute_intersection()
                
                if self.final_tip is not None:
                    # Apply smoothing
                    smoothed_final_tip = self.apply_smoothing(self.final_tip, 'final')
                    
                    if smoothed_final_tip:
                        dart_x, dart_y = smoothed_final_tip
                        final_px = mm_to_canvas_px(dart_x, dart_y)
                        
                        # Draw intersection point
                        cv2.circle(canvas, final_px, 8, (0, 0, 0), -1)  # Black outline
                        cv2.circle(canvas, final_px, 6, (0, 255, 0), -1)  # Green center
                        
                        # Calculate score and description
                        score, description = self.calculate_score(smoothed_final_tip)
                        
                        # Display position and score
                        label = f"Dart: ({dart_x:.1f}, {dart_y:.1f}) - {description}"
                        
                        cv2.putText(canvas, label, (final_px[0]+10, final_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        cv2.putText(canvas, label, (final_px[0]+10, final_px[1]),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        
                        # Print hit information to terminal
                        print(f"\nDart hit at ({dart_x:.1f}, {dart_y:.1f}) mm")
                        print(f"Distance from center: {math.sqrt(dart_x**2 + dart_y**2):.1f} mm")
                        print(f"Score: {score} - {description}")
        
        # Add calibration mode indicator if active
        if self.calibration_mode:
            cv2.putText(canvas, "CALIBRATION MODE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.calibration_point:
                cv2.putText(canvas, f"Current point: {self.calibration_point}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Highlight current calibration point on board
                cal_px = mm_to_canvas_px(*self.calibration_point)
                cv2.circle(canvas, cal_px, 10, (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]-15), (cal_px[0]+15, cal_px[1]+15), (0, 0, 255), 2)
                cv2.line(canvas, (cal_px[0]-15, cal_px[1]+15), (cal_px[0]+15, cal_px[1]-15), (0, 0, 255), 2)
        
        # Add FPS counter
        cv2.putText(canvas, f"FPS: {self.fps:.1f}", (10, canvas_size-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                  
        # Add accuracy indicator (1mm target)
        cv2.putText(canvas, "Target accuracy: 1mm", (canvas_size-250, canvas_size-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return canvas
    
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
    
    def optimize_existing_calibration(self):
        """
        Optimize the existing calibration data by:
        1. Adding interpolated points
        2. Refining calibration with polynomial fitting
        3. Validating accuracy using cross-validation
        """
        print("\nOptimizing existing calibration...")
        
        # Create manual calibration enhancer
        from manual_calibration import ManualCalibrationEnhancer
        enhancer = ManualCalibrationEnhancer(self.calibration_points)
        
        # Generate enhanced calibration points
        enhancer.generate_enhanced_calibration()
        
        # Update our calibration points with enhanced ones
        self.calibration_points = enhancer.calibration_points
        
        # Rebuild mapping tables
        self.cam1_pixel_to_board_mapping = []
        self.cam2_pixel_to_board_mapping = []
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
        
        print(f"Calibration optimized - now using {len(self.calibration_points)} points")
    
    def run(self):
        """Run the dart tracking system with optimized settings for 1mm accuracy"""
        # Initialize cameras
        cap1 = cv2.VideoCapture(self.cam_index1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        cap2 = cv2.VideoCapture(self.cam_index2)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Try to load calibration data
        try:
            self.load_calibration()
        except:
            print("Using default calibration data")
        
        # Print instructions
        print("\n*** OPTIMIZED DART TRACKING SYSTEM ***")
        print("Target accuracy: 1mm")
        print("Press 'q' to exit")
        print("Press 't' to toggle calibration mode")
        print("Press 'c' in calibration mode to capture current point")
        print("Press 'r' to reset background subtractors")
        print("Press 's' to save current calibration to file")
        print("Press 'l' to load calibration from file")
        print("Press 'o' to optimize existing calibration (for 1mm accuracy)")
        
        # FPS calculation variables
        frame_count = 0
        fps_start_time = time.time()
        
        # Main loop
        while True:
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if not ret1 or not ret2:
                print("Error reading from cameras")
                break
            
            # Process frames
            proc_frame1, fg_mask1, cam1_pixel_x = self.process_camera1_frame(frame1)
            proc_frame2, fg_mask2, cam2_pixel_x = self.process_camera2_frame(frame2)
            
            # Update board projection
            board_proj = self.update_board_projection(cam1_pixel_x, cam2_pixel_x)
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - fps_start_time
            
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = current_time
            
            # Display frames
            cv2.imshow("Camera 1", proc_frame1)
            cv2.imshow("Camera 2", proc_frame2)
            cv2.imshow("Board", board_proj)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.toggle_calibration_mode()
            elif key == ord('c') and self.calibration_mode:
                if self.calibration_point is None:
                    print("Please set calibration point first (using number keys 0-9)")
                else:
                    # Add the current detection to calibration points
                    if cam1_pixel_x is not None and cam2_pixel_x is not None:
                        print(f"Adding calibration point: {self.calibration_point} -> "
                             f"Cam1: {cam1_pixel_x:.2f}, Cam2: {cam2_pixel_x:.2f}")
                        
                        self.calibration_points[self.calibration_point] = (
                            int(round(cam1_pixel_x)), 
                            int(round(cam2_pixel_x))
                        )
                        
                        # Rebuild mapping tables
                        self.cam1_pixel_to_board_mapping = []
                        self.cam2_pixel_to_board_mapping = []
                        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
                            self.cam1_pixel_to_board_mapping.append((cam1_pixel_x, board_x))
                            self.cam2_pixel_to_board_mapping.append((cam2_pixel_x, board_y))
                        self.cam1_pixel_to_board_mapping.sort(key=lambda x: x[0])
                        self.cam2_pixel_to_board_mapping.sort(key=lambda x: x[0])
                    else:
                        print("Could not detect dart in one or both cameras")
            elif key == ord('r'):
                print("Resetting background subtractors")
                self.bg_subtractor1 = cv2.createBackgroundSubtractorMOG2(
                    history=3000, varThreshold=60, detectShadows=False
                )
                self.bg_subtractor2 = cv2.createBackgroundSubtractorMOG2(
                    history=3000, varThreshold=60, detectShadows=False
                )
                # Reset Kalman filter
                self.kalman_initialized = False
                # Reset stable position tracking
                self.stable_position_count = 0
                self.last_stable_position = None
            elif key == ord('s'):
                print("Saving calibration to file")
                self.save_calibration()
            elif key == ord('l'):
                print("Loading calibration from file")
                self.load_calibration()
            elif key == ord('o'):
                self.optimize_existing_calibration()
            elif key >= ord('0') and key <= ord('9') and self.calibration_mode:
                # Quick set calibration point using number keys
                segment_num = key - ord('0')
                if segment_num in self.board_segments:
                    self.set_calibration_point(*self.board_segments[segment_num])
                else:
                    print(f"No segment {segment_num} defined")
        
        # Cleanup
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = OptimizedDartTracker()
    tracker.run()
