# Add these to the initialization method (__init__)
self.camera_calibration_points = []  # List of (pixel_x, mm_y) tuples for calibration
# Default linear calibration as fallback
self.pixel_to_mm_factor = -0.628  # Slope in mm/pixel 
self.pixel_offset = 192.8        # Board y when pixel_x = 0

def add_calibration_point(self, pixel_x, mm_y):
    """Add a calibration point mapping pixel x-coordinate to mm y-coordinate."""
    self.camera_calibration_points.append((pixel_x, mm_y))
    # Sort by pixel_x for proper interpolation
    self.camera_calibration_points.sort(key=lambda p: p[0])
    print(f"Added calibration point: pixel_x={pixel_x}, mm_y={mm_y}")
    print(f"Current calibration points: {self.camera_calibration_points}")

def pixel_to_mm(self, pixel_x):
    """
    Convert pixel x-coordinate to mm y-coordinate using calibration points.
    Uses linear interpolation between known points or linear equation as fallback.
    """
    # If we have at least 2 calibration points, use interpolation
    if len(self.camera_calibration_points) >= 2:
        # Find the two nearest calibration points for interpolation
        for i in range(len(self.camera_calibration_points) - 1):
            p1_pixel, p1_mm = self.camera_calibration_points[i]
            p2_pixel, p2_mm = self.camera_calibration_points[i + 1]
            
            # If pixel_x is between these two points, interpolate
            if p1_pixel <= pixel_x <= p2_pixel:
                # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                return p1_mm + (pixel_x - p1_pixel) * (p2_mm - p1_mm) / (p2_pixel - p1_pixel)
                
        # If outside the calibration range, use the closest calibration point
        if pixel_x < self.camera_calibration_points[0][0]:
            return self.camera_calibration_points[0][1]
        else:
            return self.camera_calibration_points[-1][1]
    else:
        # Fallback to linear equation if not enough calibration points
        return self.pixel_to_mm_factor * pixel_x + self.pixel_offset

def clear_calibration_points(self):
    """Clear all calibration points."""
    self.camera_calibration_points = []
    print("Calibration points cleared.")
#####

dart_mm_y = self.pixel_to_mm_factor * tip_point[0] + self.pixel_offset

######

dart_mm_y = self.pixel_to_mm(tip_point[0])

#####

# Add calibration points (pixel_x, mm_y)
visualizer.add_calibration_point(100, -180)  # Leftmost point on board (-180mm) is at pixel 100
visualizer.add_calibration_point(320, 0)     # Center point (0mm) is at pixel 320
visualizer.add_calibration_point(540, 180)   # Rightmost point (180mm) is at pixel 540
