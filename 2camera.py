self.lidar1_dart.set_data(
                        [self.lidar1_projected_point[0]],
                        [self.lidar1_projected_point[1]],
                    )
                else:
                    self.lidar1_dart.set_data([], [])
            else:
                self.lidar1_vector.set_data([], [])
                self.lidar1_dart.set_data([], [])
        else:
            self.lidar1_vector.set_data([], [])
            self.lidar1_dart.set_data([], [])
            
        # Update LIDAR2 visualization
        if lidar2_most_significant is not None:
            lidar2_x, lidar2_y = lidar2_most_significant[0], lidar2_most_significant[1]

            # Draw vector from LIDAR2 position to detected point
            dx = lidar2_x - self.lidar2_pos[0]
            dy = lidar2_y - self.lidar2_pos[1]
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
                # Create 600mm vector
                unit_x = dx / length
                unit_y = dy / length
                vector_end_x = self.lidar2_pos[0] + 600 * unit_x
                vector_end_y = self.lidar2_pos[1] + 600 * unit_y

                # Draw LIDAR2 vector
                self.lidar2_vector.set_data(
                    [self.lidar2_pos[0], vector_end_x],
                    [self.lidar2_pos[1], vector_end_y],
                )

                # Draw projected point if available
                if self.lidar2_projected_point is not None:
                    self.lidar2_dart.set_data(
                        [self.lidar2_projected_point[0]],
                        [self.lidar2_projected_point[1]],
                    )
                else:
                    self.lidar2_dart.set_data([], [])
            else:
                self.lidar2_vector.set_data([], [])
                self.lidar2_dart.set_data([], [])
        else:
            self.lidar2_vector.set_data([], [])
            self.lidar2_dart.set_data([], [])

        # Update LIDAR scatter plots
        self.scatter1.set_data(x1, y1)
        self.scatter2.set_data(x2, y2)

        return (
            self.scatter1,
            self.scatter2,
            self.camera1_vector,
            self.camera2_vector,
            self.detected_dart,
            self.lidar1_vector,
            self.lidar2_vector,
            self.camera1_dart,
            self.camera2_dart,
            self.lidar1_dart,
            self.lidar2_dart,
        )

    def run(self, lidar1_script, lidar2_script):
        """Start all components."""
        lidar1_thread = threading.Thread(target=self.start_lidar, args=(lidar1_script, self.lidar1_queue, 1))
        lidar2_thread = threading.Thread(target=self.start_lidar, args=(lidar2_script, self.lidar2_queue, 2))
        
        # Make all threads daemon threads
        lidar1_thread.daemon = True
        lidar2_thread.daemon = True
        
        print("Starting LIDAR 1...")
        lidar1_thread.start()
        time.sleep(2)
        print("Starting LIDAR 2...")
        lidar2_thread.start()
        time.sleep(2)
        
        # Start camera threads last
        print("Starting cameras...")
        camera1_thread = threading.Thread(target=self.camera1_detection)
        camera2_thread = threading.Thread(target=self.camera2_detection)
        camera1_thread.daemon = True
        camera2_thread.daemon = True
        camera1_thread.start()
        time.sleep(1)
        camera2_thread.start()

        print("All sensors started. Beginning visualization...")
        anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=False, cache_frame_data=False)
        plt.show()

        # Cleanup
        self.running = False
        print("Shutting down threads...")

    def save_coefficient_scaling(self, filename="coefficient_scaling.json"):
        """Save the current coefficient scaling configuration to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.coefficient_scaling, f, indent=2)
            print(f"Coefficient scaling saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving coefficient scaling: {e}")
            return False
    
    def load_coefficient_scaling(self, filename="coefficient_scaling.json"):
        """Load coefficient scaling configuration from a JSON file."""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    loaded_scaling = json.load(f)
                    
                # Convert string keys back to integers
                self.coefficient_scaling = {int(k): v for k, v in loaded_scaling.items()}
                print(f"Coefficient scaling loaded from {filename}")
                return True
            else:
                print(f"Scaling file {filename} not found, using defaults")
                return False
        except Exception as e:
            print(f"Error loading coefficient scaling: {e}")
            return False

if __name__ == "__main__":
    lidar1_script = "./tri_test_lidar1"
    lidar2_script = "./tri_test_lidar2"
    visualizer = LidarCameraVisualizer()
    
    # Try to load coefficient scaling from file
    visualizer.load_coefficient_scaling()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            print("Calibration Mode")
            print("1. LIDAR Rotation Calibration")
            print("2. Coefficient Scaling Calibration")
            print("q. Quit")
            
            option = input("Select option: ")
            
            if option == "1":
                print("LIDAR Rotation Calibration Mode")
                print(f"Current LIDAR1 rotation: {visualizer.lidar1_rotation}°")
                print(f"Current LIDAR2 rotation: {visualizer.lidar2_rotation}°")
                
                while True:
                    cmd = input("Enter L1+/L1-/L2+/L2- followed by degrees (e.g., L1+0.5) or 'q' to quit: ")
                    if cmd.lower() == 'q':
                        break
                        
                    try:
                        if cmd.startswith("L1+"):
                            visualizer.lidar1_rotation += float(cmd[3:])
                        elif cmd.startswith("L1-"):
                            visualizer.lidar1_rotation -= float(cmd[3:])
                        elif cmd.startswith("L2+"):
                            visualizer.lidar2_rotation += float(cmd[3:])
                        elif cmd.startswith("L2-"):
                            visualizer.lidar2_rotation -= float(cmd[3:])
                            
                        print(f"Updated LIDAR1 rotation: {visualizer.lidar1_rotation}°")
                        print(f"Updated LIDAR2 rotation: {visualizer.lidar2_rotation}°")
                    except:
                        print("Invalid command format")
            elif option == "2":
                print("Coefficient Scaling Calibration Mode")
                print("Adjust scaling factors for specific segments and ring types.")
                print("Format: [segment]:[ring_type]:[scale]")
                print("  - segment: 1-20 or 'all'")
                print("  - ring_type: 'doubles', 'trebles', 'small', 'large', or 'all'")
                print("  - scale: scaling factor (e.g. 0.5, 1.0, 1.5)")
                
                while True:
                    cmd = input("Enter scaling command or 'q' to quit: ")
                    if cmd.lower() == 'q':
                        break
                        
                    try:
                        parts = cmd.split(':')
                        if len(parts) != 3:
                            print("Invalid format. Use segment:ring_type:scale")
                            continue
                            
                        segment_str, ring_type, scale_str = parts
                        scale = float(scale_str)
                        
                        # Process segment specification
                        segments = []
                        if segment_str.lower() == 'all':
                            segments = list(range(1, 21))
                        else:
                            try:
                                segment_num = int(segment_str)
                                if 1 <= segment_num <= 20:
                                    segments = [segment_num]
                                else:
                                    print("Segment must be between 1-20 or 'all'")
                                    continue
                            except ValueError:
                                print("Segment must be a number between 1-20 or 'all'")
                                continue
                        
                        # Process ring type specification
                        ring_types = []
                        if ring_type.lower() == 'all':
                            ring_types = ['doubles', 'trebles', 'small', 'large']
                        elif ring_type.lower() in ['doubles', 'trebles', 'small', 'large']:
                            ring_types = [ring_type.lower()]
                        else:
                            print("Ring type must be 'doubles', 'trebles', 'small', 'large', or 'all'")
                            continue
                        
                        # Update scaling factors
                        for segment in segments:
                            for rt in ring_types:
                                visualizer.coefficient_scaling[segment][rt] = scale
                                
                        print(f"Updated scaling factors for {len(segments)} segment(s) and {len(ring_types)} ring type(s)")
                    except ValueError:
                        print("Scale must be a numeric value")
            
            # After calibration, ask to save settings
            save = input("Save coefficient scaling settings? (y/n): ")
            if save.lower() == 'y':
                visualizer.save_coefficient_scaling()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python script.py                  - Run the program normally")
            print("  python script.py --calibrate      - Enter calibration mode")
            print("  python script.py --help           - Show this help message")
    else:
        visualizer.run(lidar1_script, lidar2_script)
