import numpy as np

def spherical_to_cartesian(r, theta, phi):
    """ Convert spherical (range, azimuth, elevation) to Cartesian (x, y, z). The angles must be in radians. """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.cos(phi)
    z = r * np.sin(phi)
    return np.array([x, y, z])

def get_rotation_matrix(roll, pitch, yaw):
    """ Creates a rotation matrix from Euler angles (radians). """
    # Rotation about X (Roll)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    # Rotation about Y (Pitch)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    # Rotation about Z (Yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def generate_fov_geometry(min_r, max_r, az_deg, el_deg):
    """ Generates the lines representing the FOV wedge in the local sensor frame of a sonar sensor. """
    az_limit = np.deg2rad(az_deg) / 2.0
    el_limit = np.deg2rad(el_deg) / 2.0
    
    corners_max = [
        spherical_to_cartesian(max_r, -az_limit, -el_limit),
        spherical_to_cartesian(max_r, -az_limit, el_limit),
        spherical_to_cartesian(max_r, az_limit, el_limit),
        spherical_to_cartesian(max_r, az_limit, -el_limit)
    ]
    
    lines = []
    # Lines that extend from min range to mex range at the four azimuth and elevation bounds
    for c in corners_max:
        p_min = c * (min_r / max_r)
        lines.append(np.array([p_min, c]))
    # Segments to break up the azimuth and elevation arcs
    az_sweep = np.linspace(-az_limit, az_limit, 20)
    el_sweep = np.linspace(-el_limit, el_limit, 20)
    # Arcs at max range that connect the lines
    lines.append(np.array([spherical_to_cartesian(max_r, az, -el_limit) for az in az_sweep]))
    lines.append(np.array([spherical_to_cartesian(max_r, az, el_limit) for az in az_sweep])) 
    lines.append(np.array([spherical_to_cartesian(max_r, -az_limit, el) for el in el_sweep]))
    lines.append(np.array([spherical_to_cartesian(max_r, az_limit, el) for el in el_sweep]))
    # Arcs at min range that connect the lines
    lines.append(np.array([spherical_to_cartesian(min_r, az, -el_limit) for az in az_sweep]))
    lines.append(np.array([spherical_to_cartesian(min_r, az, el_limit) for az in az_sweep]))
    lines.append(np.array([spherical_to_cartesian(min_r, -az_limit, el) for el in el_sweep]))
    lines.append(np.array([spherical_to_cartesian(min_r, az_limit, el) for el in el_sweep]))
    
    return lines

def generate_elevation_arc(r_feat, theta_feat, el_deg):
    """ Generates line segments that make up the elevation arc at a specified range and azimuth location (for FLS feature ambiguity). """
    el_limit = np.deg2rad(el_deg) / 2.0
    phi_vals = np.linspace(-el_limit, el_limit, 50)
    return np.array([spherical_to_cartesian(r_feat, theta_feat, phi) for phi in phi_vals])

def generate_azimuth_elevation_surface_lines(r, az_deg, el_deg, density=10):
    """ Generates grid of lines that make up the azimuth/elevation surface at a specified range (for SSS feature ambiguity) """
    az_limit = np.deg2rad(az_deg) / 2.0
    el_limit = np.deg2rad(el_deg) / 2.0
    
    lines = []
    # 1. Lines of Constant Elevation (sweeping Azimuth)
    # Generate 'density' number of horizontal arcs
    el_vals = np.linspace(-el_limit, el_limit, density)
    az_sweep = np.linspace(-az_limit, az_limit, 50)
    for phi in el_vals:
        # Create an arc for this fixed elevation
        arc = np.array([spherical_to_cartesian(r, theta, phi) for theta in az_sweep])
        lines.append(arc)
    # 2. Lines of Constant Azimuth (sweeping Elevation)
    # Generate 'density' number of vertical arcs
    az_vals = np.linspace(-az_limit, az_limit, density)
    el_sweep = np.linspace(-el_limit, el_limit, 50)
    for theta in az_vals:
        # Create an arc for this fixed azimuth
        arc = np.array([spherical_to_cartesian(r, theta, phi) for phi in el_sweep])
        lines.append(arc)
    return lines