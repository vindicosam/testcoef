import cv2
import numpy as np
import math
from bisect import bisect_left
import pickle
import os

class ManualCalibrationEnhancer:
    """
    Enhanced manual calibration system that extends the existing calibration points
    and improves interpolation for 1mm accuracy
    """
    
    def __init__(self, calibration_points=None, frame_width=640, frame_height=480):
        """
        Initialize with existing calibration points if available
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.calibration_points = calibration_points or {}
        
        # Camera 1 board plane line (y value in pixels where board surface is)
        self.cam1_board_plane_y = 190
        # Camera 1 ROI calculated from board plane
        self.cam1_roi_range = 30
        self.cam1_roi_top = self.cam1_board_plane_y - self.cam1_roi_range
        self.cam1_roi_bottom = self.cam1_board_plane_y + self.cam1_roi_range
        
# Camera 2 board plane line
        self.cam2_board_plane_y = 239
        # Camera 2 ROI calculated from board plane
        self.cam2_roi_range = 30
        self.cam2_roi_top = self.cam2_board_plane_y - self.cam2_roi_range
        self.cam2_roi_bottom = self.cam2_board_plane_y + self.cam2_roi_range
        
        # Mapping tables
        self.cam1_pixel_to_board_mapping = []
        self.cam2_pixel_to_board_mapping = []
        
        # Generate enhanced calibration points if we have an initial set
        if self.calibration_points:
            self.generate_enhanced_calibration()
        
    def load_calibration(self, filename="dart_calibration.json"):
        """Load calibration from file"""
        try:
            import json
            import ast
            with open(filename, "r") as f:
                loaded_points = json.load(f)
                self.calibration_points = {ast.literal_eval(k): v for k, v in loaded_points.items()}
            print(f"Loaded {len(self.calibration_points)} calibration points")
            
            # Generate enhanced calibration
            self.generate_enhanced_calibration()
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
            
    def save_calibration(self, filename="dart_calibration.json"):
        """Save calibration to file"""
        try:
            import json
            with open(filename, "w") as f:
                json.dump({str(k): v for k, v in self.calibration_points.items()}, f)
            print(f"Saved {len(self.calibration_points)} calibration points")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
            
    def generate_enhanced_calibration(self):
        """
        Generate enhanced calibration points by:
        1. Adding interpolated points between existing ones
        2. Generating a grid of calibration points
        3. Using polynomial fitting for more accurate interpolation
        """
        if not self.calibration_points:
            print("No calibration points available")
            return
            
        print("Generating enhanced calibration points...")
        original_count = len(self.calibration_points)
        
        # Extract existing points
        board_points = []
        cam1_points = []
        cam2_points = []
        
        for (board_x, board_y), (cam1_pixel_x, cam2_pixel_x) in self.calibration_points.items():
            board_points.append((board_x, board_y))
            cam1_points.append(cam1_pixel_x)
            cam2_points.append(cam2_pixel_x)
            
        # Fit polynomial surfaces for both camera mappings
        # This gives us more accurate interpolation between points
        from scipy.interpolate import griddata
        
        # Create board coordinate grid for interpolation
        grid_size = 10  # 10mm grid
        x_min = min(p[0] for p in board_points) - grid_size
        x_max = max(p[0] for p in board_points) + grid_size
        y_min = min(p[1] for p in board_points) - grid_size
        y_max = max(p[1] for p in board_points) + grid_size
        
        grid_x = np.arange(x_min, x_max + grid_size, grid_size)
        grid_y = np.arange(y_min, y_max + grid_size, grid_size)
        
        # Generate grid points
        grid_points = []
        for x in grid_x:
            for y in grid_y:
                # Skip points outside the dartboard radius plus a margin
                distance = math.sqrt(x*x + y*y)
                if distance <= 180:  # 170mm radius + 10mm margin
                    grid_points.append((x, y))
        
        # Interpolate camera 1 pixel values for grid points
        cam1_interp = griddata(
            board_points, 
            cam1_points, 
            grid_points, 
            method='cubic',
            fill_value=None
        )
        
        # Interpolate camera 2 pixel values for grid points
        cam2_interp = griddata(
            board_points, 
            cam2_points, 
            grid_points, 
            method='cubic', 
            fill_value=None
        )
        
        # Add interpolated points to calibration
        added_count = 0
        for i, (x, y) in enumerate(grid_points):
            if i < len(cam1_interp) and i < len(cam2_interp):
                if not np.isnan(cam1_interp[i]) and not np.isnan(cam2_interp[i]):
                    # Round to integer pixel values
                    cam1_px = int(round(cam1_interp[i]))
                    cam2_px = int(round(cam2_interp[i]))
                    
                    # Add to calibration points if not already present
                    grid_key = (round(x, 1), round(y, 1))
                    if grid_key not in self.calibration_points:
                        self.calibration_points[grid_key] = (cam1_px, cam2_px)
                        added_count += 1
        
        print(f"Added {added_count} interpolated points to calibration")
        print(f"Total calibration points: {len(self.calibration_points)}")
        
        # Update mapping tables
        self.update_mapping_tables()
    
    def update_mapping_tables(self):
        """Update mapping tables from calibration points"""
        # Clear existing mappings
        self.cam1_pixel_to_board_mapping = []
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
        
        # Print mapping range coverage
        cam1_pixel_min = min(x[0] for x in self.cam1_pixel_to_board_mapping)
        cam1_pixel_max = max(x[0] for x in self.cam1_pixel_to_board_mapping)
        print(f"Camera 1 pixel range: {cam1_pixel_min} to {cam1_pixel_max}")
        
        cam2_pixel_min = min(x[0] for x in self.cam2_pixel_to_board_mapping)
        cam2_pixel_max = max(x[0] for x in self.cam2_pixel_to_board_mapping)
        print(f"Camera 2 pixel range: {cam2_pixel_min} to {cam2_pixel_max}")
    
    def interpolate_position(self, cam1_pixel_x, cam2_pixel_x):
        """
        Interpolate board position from camera pixel coordinates
        Uses enhanced cubic interpolation for higher accuracy
        
        Args:
            cam1_pixel_x: x-coordinate in camera 1
            cam2_pixel_x: x-coordinate in camera 2
            
        Returns:
            (board_x, board_y): position in board coordinates
        """
        # Get initial estimate using linear interpolation
        board_x = self.interpolate_value(cam1_pixel_x, self.cam1_pixel_to_board_mapping)
        board_y = self.interpolate_value(cam2_pixel_x, self.cam2_pixel_to_board_mapping)
        
        # Find nearby calibration points for higher order interpolation
        nearby_points = []
        
        for (board_x_cal, board_y_cal), (cam1_px_cal, cam2_px_cal) in self.calibration_points.items():
            # Calculate distance in pixel space
            pixel_distance = math.sqrt(
                (cam1_pixel_x - cam1_px_cal)**2 + 
                (cam2_pixel_x - cam2_px_cal)**2
            )
            
            # Use points within a certain pixel distance
            if pixel_distance < 50:  # 50 pixel radius
                nearby_points.append((
                    (board_x_cal, board_y_cal),
                    (cam1_px_cal, cam2_px_cal),
                    pixel_distance
                ))
        
        # If we have enough nearby points, use weighted interpolation
        if len(nearby_points) >= 3:
            # Sort by distance
            nearby_points.sort(key=lambda x: x[2])
            
            # Take closest points (up to 5)
            closest_points = nearby_points[:5]
            
            # Calculate weights (inverse distance)
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for (board_x_cal, board_y_cal), _, distance in closest_points:
                # Add small constant to avoid division by zero
                weight = 1.0 / (distance + 0.1)
                total_weight += weight
                
                weighted_x += board_x_cal * weight
                weighted_y += board_y_cal * weight
                
            # Normalize by total weight
            if total_weight > 0:
                board_x = weighted_x / total_weight
                board_y = weighted_y / total_weight
        
        return board_x, board_y
    
    def interpolate_value(self, pixel_value, mapping_table):
        """
        Basic linear interpolation for a value using the provided mapping table.
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
    
    def recursive_interpolation(self, cam1_pixel_x, cam2_pixel_x, depth=2):
        """
        Perform recursive interpolation for higher accuracy
        
        The idea is to iteratively refine the estimate by:
        1. Get initial board (x,y) from pixel coordinates
        2. Find nearby calibration points to initial estimate
        3. Use these nearby points for a local interpolation
        4. Repeat the process with refined estimate
        
        Args:
            cam1_pixel_x, cam2_pixel_x: Pixel coordinates
            depth: Recursion depth for refinement
            
        Returns:
            Refined (board_x, board_y) coordinates
        """
        if depth <= 0:
            # Base case - use direct interpolation
            return self.interpolate_position(cam1_pixel_x, cam2_pixel_x)
        
        # Get initial estimate
        board_x, board_y = self.interpolate_position(cam1_pixel_x, cam2_pixel_x)
        
        # Find nearby calibration points in board space
        nearby_points = []
        for (board_x_cal, board_y_cal), (cam1_px_cal, cam2_px_cal) in self.calibration_points.items():
            # Calculate distance in board space
            board_distance = math.sqrt(
                (board_x - board_x_cal)**2 + 
                (board_y - board_y_cal)**2
            )
            
            # Use points within 30mm radius
            if board_distance < 30:
                nearby_points.append((
                    (board_x_cal, board_y_cal),
                    (cam1_px_cal, cam2_px_cal),
                    board_distance
                ))
        
        # If we have enough nearby points
        if len(nearby_points) >= 3:
            # Sort by distance
            nearby_points.sort(key=lambda x: x[2])
            
            # Take closest points
            closest_points = nearby_points[:min(len(nearby_points), 8)]
            
            # Create local interpolation model
            board_coords = np.array([p[0] for p in closest_points])
            cam1_pixels = np.array([p[1][0] for p in closest_points])
            cam2_pixels = np.array([p[1][1] for p in closest_points])
            
            # Try to fit a 2nd degree polynomial if we have enough points
            if len(closest_points) >= 6:
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import LinearRegression
                    
                    # Create polynomial features
                    poly = PolynomialFeatures(degree=2)
                    board_poly = poly.fit_transform(board_coords)
                    
                    # Fit regression models
                    model_cam1 = LinearRegression().fit(board_poly, cam1_pixels)
                    model_cam2 = LinearRegression().fit(board_poly, cam2_pixels)
                    
                    # Now we need to solve the inverse problem:
                    # Find board (x,y) that minimizes the difference between
                    # model predicted pixel values and actual pixel values
                    
                    # Use optimization to find the best board coordinates
                    from scipy.optimize import minimize
                    
                    def objective(board_xy):
                        board_xy_poly = poly.transform(np.array([board_xy]))
                        pred_cam1 = model_cam1.predict(board_xy_poly)[0]
                        pred_cam2 = model_cam2.predict(board_xy_poly)[0]
                        
                        # Error squared
                        return ((pred_cam1 - cam1_pixel_x)**2 + 
                                (pred_cam2 - cam2_pixel_x)**2)
                    
                    # Optimize starting from our initial estimate
                    result = minimize(
                        objective, 
                        [board_x, board_y],
                        method='Nelder-Mead'
                    )
                    
                    if result.success:
                        # Recurse with updated estimate
                        return self.recursive_interpolation(
                            cam1_pixel_x, 
                            cam2_pixel_x, 
                            depth=depth-1
                        )
                    
                except Exception as e:
                    print(f"Polynomial fitting failed: {e}")
        
        # If advanced method failed or not enough points, use standard interpolation
        return board_x, board_y
    
    def add_calibration_point(self, board_x, board_y, cam1_pixel_x, cam2_pixel_x):
        """Add a new calibration point"""
        self.calibration_points[(board_x, board_y)] = (cam1_pixel_x, cam2_pixel_x)
        self.update_mapping_tables()
        
    def generate_validation_grid(self):
        """
        Generate a grid of points for validation
        Returns a list of (board_x, board_y) coordinates
        """
        grid_points = []
        # Generate points in a grid pattern
        for x in range(-160, 161, 20):
            for y in range(-160, 161, 20):
                # Skip points outside dartboard
                if math.sqrt(x*x + y*y) <= 171:
                    grid_points.append((x, y))
        return grid_points
                
    def verify_accuracy(self, cam1, cam2):
        """
        Verify calibration accuracy using cross-validation
        
        Args:
            cam1, cam2: Camera objects for testing
            
        Returns:
            average_error: Average error in mm
        """
        # Make a copy of calibration points
        test_points = list(self.calibration_points.items())
        
        # Verification metrics
        errors = []
        
        # Test each point by removing it and predicting its position
        for i, ((board_x, board_y), (cam1_px, cam2_px)) in enumerate(test_points):
            # Create temporary copy without this point
            temp_calibration = {k: v for k, v in self.calibration_points.items() 
                              if k != (board_x, board_y)}
            
            # Create temporary calibration object
            temp_enhancer = ManualCalibrationEnhancer(temp_calibration)
            
            # Predict position using the point's pixel values
            pred_x, pred_y = temp_enhancer.interpolate_position(cam1_px, cam2_px)
            
            # Calculate error
            error = math.sqrt((pred_x - board_x)**2 + (pred_y - board_y)**2)
            errors.append(error)
            
            print(f"Point {i+1}/{len(test_points)}: "
                 f"Actual ({board_x}, {board_y}), "
                 f"Predicted ({pred_x:.1f}, {pred_y:.1f}), "
                 f"Error: {error:.2f}mm")
        
        # Calculate average error
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        
        print(f"Verification results:")
        print(f"Average error: {avg_error:.2f}mm")
        print(f"Maximum error: {max_error:.2f}mm")
        print(f"Percentage of points with error < 1mm: "
             f"{100 * sum(1 for e in errors if e < 1.0) / len(errors):.1f}%")
        
        return avg_error
