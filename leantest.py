import cv2
import numpy as np
import sys
import signal

class DartTipDetector:
    def __init__(self):
        # ROI settings
        self.roi_top = 148
        self.roi_bottom = 185
        
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
    
    def detect_dart_tip(self, mask):
        """Detect the dart tip position from binary mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # Find the highest point (lowest y-value) across all contours
        # This is the tip of the dart
        tip_point = None
        for contour in contours:
            if cv2.contourArea(contour) < 20:  # Filter out noise
                continue
                
            for point in contour:
                x, y = point[0]
                if tip_point is None or y < tip_point[1]:
                    tip_point = (x, y)
        
        return tip_point, contours
    
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
                
                # Create a copy of ROI for visualization
                roi_vis = roi.copy()
                
                # Detect dart tip
                tip_point, contours = self.detect_dart_tip(thresh)
                
                # Draw all contours
                if contours:
                    cv2.drawContours(roi_vis, contours, -1, (0, 255, 0), 2)
                
                # Process if tip detected
                if tip_point is not None:
                    # Mark the tip point
                    cv2.circle(roi_vis, tip_point, 5, (0, 0, 255), -1)
                    
                    # Measure angle
                    angle_info = self.measure_tip_angle(thresh, tip_point)
                    
                    if angle_info is not None:
                        angle, lean, points_below = angle_info
                        
                        # Draw all detected points
                        for point in points_below:
                            cv2.circle(roi_vis, point, 1, (255, 0, 255), -1)
                        
                        # Draw the line fitted through these points
                        if len(points_below) > 1:
                            x_values = [p[0] for p in points_below]
                            y_values = [p[1] for p in points_below]
                            min_x, max_x = min(x_values), max(x_values)
                            
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
                        
                        # Display angle info on frame
                        text = f"Angle: {angle:.1f}° (Avg: {angle:.1f}°) | Lean: {lean}"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Display on ROI too
                        roi_text = f"Angle: {angle:.1f}° ({lean})"
                        cv2.putText(roi_vis, roi_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Copy ROI visualization back to the main frame
                frame[self.roi_top:self.roi_bottom, :] = roi_vis
                
                # Show the mask for debugging
                mask_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_rgb, "Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display the mask below the ROI
                frame_with_mask = np.copy(frame)
                mask_height = self.roi_bottom - self.roi_top
                mask_y_offset = self.roi_bottom + 10
                
                # Ensure the mask fits within the frame
                if mask_y_offset + mask_height < frame.shape[0]:
                    frame_with_mask[mask_y_offset:mask_y_offset+mask_height, 0:mask_rgb.shape[1]] = mask_rgb
                
                # Display the final frame
                cv2.imshow('Dart Tip Angle Detection', frame_with_mask)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released.")

if __name__ == "__main__":
    detector = DartTipDetector()
    detector.run()
