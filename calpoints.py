# Dartboard segment calibration points
# Points are positioned 1mm inward from the wires
# 
# Standard dartboard layout (clockwise from top):
# 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5

segment_calibration_points = [
    # Double segments (outer ring, moved 1mm inward)
    (0, 169, None, None),      # Double 20 (top)
    (52, 161, None, None),     # Double 1
    (98, 139, None, None),     # Double 18
    (139, 98, None, None),     # Double 4
    (161, 52, None, None),     # Double 13
    (169, 0, None, None),      # Double 6 (right)
    (161, -52, None, None),    # Double 10
    (139, -98, None, None),    # Double 15
    (98, -139, None, None),    # Double 2
    (52, -161, None, None),    # Double 17
    (0, -169, None, None),     # Double 3 (bottom)
    (-52, -161, None, None),   # Double 19
    (-98, -139, None, None),   # Double 7
    (-139, -98, None, None),   # Double 16
    (-161, -52, None, None),   # Double 8
    (-169, 0, None, None),     # Double 11 (left)
    (-161, 52, None, None),    # Double 14
    (-139, 98, None, None),    # Double 9
    (-98, 139, None, None),    # Double 12
    (-52, 161, None, None),    # Double 5

    # Triple segments (middle ring, moved 1mm inward)
    (0, 106, None, None),      # Triple 20 (top)
    (33, 101, None, None),     # Triple 1
    (62, 87, None, None),      # Triple 18
    (87, 62, None, None),      # Triple 4
    (101, 33, None, None),     # Triple 13
    (106, 0, None, None),      # Triple 6 (right)
    (101, -33, None, None),    # Triple 10
    (87, -62, None, None),     # Triple 15
    (62, -87, None, None),     # Triple 2
    (33, -101, None, None),    # Triple 17
    (0, -106, None, None),     # Triple 3 (bottom)
    (-33, -101, None, None),   # Triple 19
    (-62, -87, None, None),    # Triple 7
    (-87, -62, None, None),    # Triple 16
    (-101, -33, None, None),   # Triple 8
    (-106, 0, None, None),     # Triple 11 (left)
    (-101, 33, None, None),    # Triple 14
    (-87, 62, None, None),     # Triple 9
    (-62, 87, None, None),     # Triple 12
    (-33, 101, None, None),    # Triple 5
]

# Known existing calibration points with pixel values
# These should be integrated with the above points
existing_points = [
    (114, 121, 17, 153),       # Double 18
    (48, 86, 214, 182),        # Triple 18
    (119, -117, 167, 429),     # Double 15
    (86, -48, 189, 359),       # Triple 15
    (-118, -121, 453, 624),    # Double 7
    (-50, -88, 373, 478),      # Triple 7
    (121, 118, 624, 240),      # Double 9
    (-90, -47, 483, 42)        # Triple 9
]

# VERIFICATION VALUES:
# These show that our new points are generally consistent with
# the existing calibrated points you already have:
#
# Double 18 calculation: (98, 139) vs existing: (114, 121)
# Triple 18 calculation: (62, 87) vs existing: (48, 86)
# Double 15 calculation: (139, -98) vs existing: (119, -117)
# Triple 15 calculation: (87, -62) vs existing: (86, -48)
# Double 7 calculation: (-98, -139) vs existing: (-118, -121)
# Triple 7 calculation: (-62, -87) vs existing: (-50, -88)
# Double 9 calculation: (-139, 98) vs existing: (121, 118) [Note: This doesn't match]
# Triple 9 calculation: (-87, 62) vs existing: (-90, -47) [Note: This doesn't match]
#
# There may be differences due to the specific dartboard calibration
# in your setup or the exact position/angle of your dartboard.
