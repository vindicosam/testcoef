import cv2
import numpy as np
import time
import signal
import sys

class SimpleDartDetector:
    def __init__(self):
        # Camera configuration
        self.roi_top = 148  # Top of the ROI
        self.roi_bottom = 185  # Bottom of the ROI
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )
        
        # Running flag
        self.running = True
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        sys.exit(0)
        
    def detect_dart_angle(self, contour):
        """
        Find the angle of the dart by fitting an ellipse to the contour.
        """
        if len(contour) < 5:
            return None, None
            
        try:
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # In OpenCV, the angle is measured counter-clockwise from the horizontal axis
            # We need to adjust it to be relative to vertical for our application
            
            # Convert to our reference system where:
            # 90° = vertical
            # <90° = leaning left
            # >90° = leaning right
            if angle > 90:
                adjusted_angle = 180 - angle
            else:
                adjusted_angle = angle
                
            return center, adjusted_angle
            
        except Exception as e:
            print(f"Error fitting ellipse: {e}")
            return None, None
    
    def run(self):
        """Main processing loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
            
        print("Camera opened successfully. Press 'q' to exit.")
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break
                
                # Rotate the frame 180 degrees (if camera is upside down)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Draw ROI rectangle
                cv2.rectangle(frame, (0, self.roi_top), (frame.shape[1], self.roi_bottom), (0, 255, 0), 2)
                
                # Extract ROI
                roi = frame[self.roi_top:self.roi_bottom, :]
                
                # Apply background subtraction
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                fg_mask = self.bg_subtractor.apply(gray)
                
                # Threshold and clean up the mask
                _, thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=1)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create a copy of ROI for visualization
                roi_vis = roi.copy()
                
                # Find the largest contour (assumed to be the dart)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Only process if contour is large enough
                    if cv2.contourArea(largest_contour) > 30:
                        # Draw contour
                        cv2.drawContours(roi_vis, [largest_contour], -1, (0, 255, 0), 2)
                        
                        # Get dart center and angle
                        center, angle = self.detect_dart_angle(largest_contour)
                        
                        if center is not None and angle is not None:
                            # Calculate the center point adjusted to full frame coordinates
                            center_x, center_y = int(center[0]), int(center[1])
                            frame_center_y = center_y + self.roi_top
                            
                            # Draw the center point
                            cv2.circle(roi_vis, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            # Calculate line endpoints based on the angle
                            line_length = 50
                            
                            # Calculate the deviation from vertical
                            deviation_deg = 90 - angle  # positive = leaning left, negative = leaning right
                            deviation_rad = np.radians(deviation_deg)
                            
                            # Calculate line endpoints
                            dx = line_length * np.sin(deviation_rad)
                            dy = line_length * np.cos(deviation_rad)
                            
                            # Calculate endpoints
                            start_x = int(center_x - dx)
                            start_y = int(center_y - dy)
                            end_x = int(center_x + dx)
                            end_y = int(center_y + dy)
                            
                            # Draw the line through the dart
                            cv2.line(roi_vis, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
                            
                            # Determine lean direction
                            lean_direction = "Vertical"
                            if angle < 85:
                                lean_direction = "Left"
                            elif angle > 95:
                                lean_direction = "Right"
                                
                            # Add text showing the angle
                            cv2.putText(roi_vis, f"Angle: {angle:.1f}° ({lean_direction})", 
                                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            # Also draw on the full frame
                            frame_start_x = start_x
                            frame_start_y = start_y + self.roi_top
                            frame_end_x = end_x
                            frame_end_y = end_y + self.roi_top
                            
                            # Draw extended line on full frame
                            extension_factor = 3.0
                            full_dx = dx * extension_factor
                            full_dy = dy * extension_factor
                            frame_ext_start_x = int(center_x - full_dx)
                            frame_ext_start_y = int(center_y - full_dy) + self.roi_top
                            frame_ext_end_x = int(center_x + full_dx)
                            frame_ext_end_y = int(center_y + full_dy) + self.roi_top
                            
                            cv2.line(frame, (frame_ext_start_x, frame_ext_start_y), 
                                    (frame_ext_end_x, frame_ext_end_y), (0, 0, 255), 2)
                            
                            # Add text to main frame
                            cv2.putText(frame, f"Angle: {angle:.1f}° ({lean_direction})", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Copy ROI visualization back to the main frame
                frame[self.roi_top:self.roi_bottom, :] = roi_vis
                
                # Also show the mask for debugging
                mask_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                resized_mask = cv2.resize(mask_rgb, (320, 37))  # Half width, same height as ROI
                frame[self.roi_bottom+10:self.roi_bottom+10+resized_mask.shape[0], 
                      10:10+resized_mask.shape[1]] = resized_mask
                
                # Display the frame
                cv2.imshow('Dart Angle Detection', frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released.")

if __name__ == "__main__":
    detector = SimpleDartDetector()
    detector.run()
