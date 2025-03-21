import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import signal
import sys
import math

class SimpleCameraLeanDetector:
    def __init__(self):
        # Camera configuration
        self.camera_data = {"dart_mm_x": None, "dart_angle": None}
        
        # ROI Settings and Pixel-to-mm Mapping
        self.roi_top = 148  # Top of the ROI
        self.roi_bottom = 185  # Bottom of the ROI
        self.pixel_to_mm_x = (180 - (-180)) / (556 - 126)  # Calibrated conversion
        self.camera_x_offset = 9.8  # Small offset to account for systematic error
        
        # Detection persistence to maintain visibility
        self.last_valid_detection = {"dart_mm_x": None, "dart_angle": None}
        self.detection_persistence_counter = 0
        self.detection_persistence_frames = 30
        
        # Camera background subtractor with sensitive parameters
        self.camera_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=25, detectShadows=False
        )
        
        # 3D lean detection variables
        self.current_side_lean_angle = 90.0  # 90 degrees = vertical
        self.lean_history = []  # Store recent lean readings for smoothing
        self.max_lean_history = 10  # Keep track of last 10 lean readings
        
        # Running flag
        self.running = True
        
        # Setup visualization
        self.setup_plot()
        
        # Signal handling
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        self.running = False
        print("\nShutting down...")
        plt.close("all")
        sys.exit(0)
    
    def setup_plot(self):
        """Initialize the visualization plot"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.suptitle("Dart Lean Detection", fontsize=16)
        
        # Create subplot layout
        self.gs = self.fig.add_gridspec(2, 2)
        
        # Camera view
        self.ax_camera = self.fig.add_subplot(self.gs[0, 0])
        self.ax_camera.set_title("Camera View")
        self.camera_img = self.ax_camera.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # ROI view
        self.ax_roi = self.fig.add_subplot(self.gs[0, 1])
        self.ax_roi.set_title("ROI with Detected Dart")
        self.roi_img = self.ax_roi.imshow(np.zeros((self.roi_bottom - self.roi_top, 640, 3), dtype=np.uint8))
        
        # Processed mask view
        self.ax_mask = self.fig.add_subplot(self.gs[1, 0])
        self.ax_mask.set_title("Processed Mask")
        self.mask_img = self.ax_mask.imshow(np.zeros((self.roi_bottom - self.roi_top, 640), dtype=np.uint8), cmap='gray')
        
        # Lean visualization
        self.ax_lean = self.fig.add_subplot(self.gs[1, 1])
        self.ax_lean.set_title("Dart Lean Visualization")
        self.ax_lean.set_xlim(-2, 2)
        self.ax_lean.set_ylim(-0.5, 2.5)
        self.ax_lean.grid(True)
        self.ax_lean.set_xlabel("Left-Right Position")
        self.ax_lean.set_ylabel("Height")
        
        # Draw reference lines for lean
        self.ax_lean.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Ground line
        self.ax_lean.axvline(x=0, color='black', linestyle='--', alpha=0.3)  # Center line
        
        # Dart line visualization
        self.dart_line, = self.ax_lean.plot([], [], 'r-', linewidth=3)
        
        # Text display for measurements
        self.lean_text = self.ax_lean.text(-1.9, 2.2, "", fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
    
    def calculate_dart_angle(self, contour):
        """
        Calculate the angle of the dart tip relative to vertical.
        Returns angle in degrees where:
        - 90 degrees = perfectly upright (perpendicular to board)
        - 0 degrees = flat against the board (parallel)
        """
        if len(contour) < 5:
            return None
        
        # Fit an ellipse to the contour
        try:
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Convert to 0-90 degrees relative to vertical (90° = vertical)
            if angle > 90:
                angle = 180 - angle
            
            return angle
        except:
            return None
    
    def process_camera_frame(self, frame):
        """
        Process a single camera frame to detect dart and lean angle.
        
        Args:
            frame: Input camera frame
            
        Returns:
            processed_frame: Frame with detection overlay
            roi: Region of interest showing dart
            mask: Processed binary mask 
            detection_data: Dictionary with detection results
        """
        # Make a copy for visualization
        vis_frame = frame.copy()
        
        # Flip the frame 180 degrees since camera is upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        vis_frame = cv2.rotate(vis_frame, cv2.ROTATE_180)
        
        # Extract ROI
        roi = frame[self.roi_top:self.roi_bottom, :]
        roi_vis = roi.copy()
        
        # Background subtraction and thresholding
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        fg_mask = self.camera_bg_subtractor.apply(gray)
        
        # More sensitive threshold
        fg_mask = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to enhance the dart
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        
        # Reset current detection
        detection_data = {
            "dart_mm_x": None,
            "dart_angle": None,
            "tip_point": None,
            "contour": None
        }
        
        # Detect contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the dart tip (highest point since image is flipped)
            tip_contour = None
            lowest_point = (-1, -1)
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Reduced threshold for smaller darts
                    for point in contour:
                        x, y = point[0]
                        if tip_contour is None or y < lowest_point[1]:
                            lowest_point = (x, y)
                            tip_contour = contour
            
            if tip_contour is not None:
                # Get dart angle
                dart_angle = self.calculate_dart_angle(tip_contour)
                
                # Map pixels to mm coordinates with corrected mapping
                tip_pixel_x = lowest_point[0]
                dart_mm_x = 180 - (tip_pixel_x - 126) * self.pixel_to_mm_x + self.camera_x_offset
                
                # Save detection data
                detection_data["dart_mm_x"] = dart_mm_x
                detection_data["dart_angle"] = dart_angle
                detection_data["tip_point"] = lowest_point
                detection_data["contour"] = tip_contour
                
                # Update persistence
                self.last_valid_detection["dart_mm_x"] = dart_mm_x
                self.last_valid_detection["dart_angle"] = dart_angle
                self.detection_persistence_counter = self.detection_persistence_frames
                
                # Draw detection on visualization
                cv2.drawContours(roi_vis, [tip_contour], -1, (0, 255, 0), 2)
                cv2.circle(roi_vis, lowest_point, 5, (0, 0, 255), -1)
                
                # Draw lean angle indicator
                if dart_angle is not None:
                    # Calculate line endpoint for angle visualization
                    line_length = 30
                    angle_rad = np.radians(90 - dart_angle)  # Convert to radians and adjust
                    end_x = int(lowest_point[0] + line_length * np.sin(angle_rad))
                    end_y = int(lowest_point[1] + line_length * np.cos(angle_rad))
                    
                    # Draw line showing dart angle
                    cv2.line(roi_vis, lowest_point, (end_x, end_y), (255, 0, 0), 2)
                    
                    # Add angle text
                    cv2.putText(roi_vis, f"{dart_angle:.1f}°", 
                               (lowest_point[0] + 10, lowest_point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # If no dart detected but we have a valid previous detection
        elif self.detection_persistence_counter > 0:
            self.detection_persistence_counter -= 1
            if self.detection_persistence_counter > 0:
                detection_data["dart_mm_x"] = self.last_valid_detection["dart_mm_x"]
                detection_data["dart_angle"] = self.last_valid_detection["dart_angle"]
        
        # Draw ROI box on the main frame
        cv2.rectangle(vis_frame, (0, self.roi_top), (vis_frame.shape[1], self.roi_bottom), (255, 0, 0), 2)
        
        # Add detected information to the frame
        if detection_data["dart_mm_x"] is not None:
            text = f"X: {detection_data['dart_mm_x']:.1f}mm"
            cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        if detection_data["dart_angle"] is not None:
            text = f"Angle: {detection_data['dart_angle']:.1f}°"
            cv2.putText(vis_frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame, roi_vis, fg_mask, detection_data
    
    def visualize_lean(self, dart_angle, dart_mm_x):
        """
        Update the lean visualization based on detected angle and position.
        
        Args:
            dart_angle: Angle of dart in degrees (90° = vertical)
            dart_mm_x: X position of dart in mm
        """
        if dart_angle is None or dart_mm_x is None:
            # No valid data, show no dart
            self.dart_line.set_data([], [])
            self.lean_text.set_text("No dart detected")
            return
        
        # Normalize position to -2 to 2 range
        normalized_pos = dart_mm_x / 180.0  # Assuming dart_mm_x ranges from -180 to 180
        
        # Calculate dart line points
        dart_length = 2.0  # Length of visualization line
        
        # Convert to radians and adjust coordinate system
        angle_rad = np.radians(90 - dart_angle)  # 90° = vertical
        
        # Calculate endpoint
        x_end = normalized_pos + dart_length * np.sin(angle_rad)
        y_end = dart_length * np.cos(angle_rad)
        
        # Update dart line visualization
        self.dart_line.set_data([normalized_pos, x_end], [0, y_end])
        
        # Interpret lean direction
        lean_direction = "Vertical"
        if dart_angle < 85:
            lean_direction = "Leaning Left"
        elif dart_angle > 95:
            lean_direction = "Leaning Right"
        
        # Update text
        self.lean_text.set_text(
            f"Position: {dart_mm_x:.1f} mm\n"
            f"Angle: {dart_angle:.1f}° ({lean_direction})\n"
            f"Deviation: {abs(90-dart_angle):.1f}° from vertical"
        )
    
    def update_plot(self, frame_data):
        """Update the visualization with new frame data"""
        frame, roi, mask, detection = frame_data
        
        # Update images
        self.camera_img.set_array(frame)
        self.roi_img.set_array(roi)
        self.mask_img.set_array(mask)
        
        # Update lean visualization
        self.visualize_lean(detection["dart_angle"], detection["dart_mm_x"])
        
        return self.camera_img, self.roi_img, self.mask_img, self.dart_line, self.lean_text
    
    def camera_loop(self):
        """Main camera processing loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Make sure the camera is opened
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Camera opened successfully. Press Ctrl+C to exit.")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break
                
                # Process the frame
                processed_frame, roi, mask, detection = self.process_camera_frame(frame)
                
                # Update camera data
                self.camera_data["dart_mm_x"] = detection["dart_mm_x"]
                self.camera_data["dart_angle"] = detection["dart_angle"]
                
                # Log data for debugging
                if detection["dart_angle"] is not None:
                    # Add to lean history for smoothing
                    self.lean_history.append(detection["dart_angle"])
                    if len(self.lean_history) > self.max_lean_history:
                        self.lean_history.pop(0)
                    
                    # Calculate average lean
                    avg_lean = sum(self.lean_history) / len(self.lean_history)
                    self.current_side_lean_angle = avg_lean
                    
                    # Print information for debugging
                    print(f"Position: {detection['dart_mm_x']:.1f}mm, "
                          f"Angle: {detection['dart_angle']:.1f}° (avg: {avg_lean:.1f}°)")
                
                # Update the plot with new data
                self.update_plot((processed_frame, roi, mask, detection))
                plt.pause(0.01)  # Give the GUI time to update
        
        finally:
            cap.release()
            print("Camera released.")
    
    def run(self):
        """Start the detector"""
        # Start camera processing
        self.camera_loop()

if __name__ == "__main__":
    detector = SimpleCameraLeanDetector()
    detector.run()
