import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
from mpl_toolkits.mplot3d import Axes3D

import utils as ut

def solve_fls_fls(meas1, meas2, R_rel, t_rel, elevation_limit_deg=20.0):
    """
    Recover 3D feature position from two FLS observations.
    Returns worst_estimate, and a list of all geometrically valid candidates.
    """
    r1, theta1 = meas1
    r2, theta2 = meas2
    
    t = np.array(t_rel).flatten()
    R = np.array(R_rel)
    
    # 1. Compute Coefficients
    u = t.T @ R
    ux, uy, uz = u
    
    A = r1 * (ux * np.cos(theta1) + uy * np.sin(theta1))
    B = r1 * uz
    C = (r2**2 - r1**2 - np.linalg.norm(t)**2) / 2.0
    
    # 2. Check Existence
    D = np.sqrt(A**2 + B**2)
    if D == 0: return None, [] # Degeneracy
        
    cos_arg = C / D
    if np.abs(cos_arg) > 1.0: return None, [] # Non-intersection
        
    # 3. Harmonic Addition
    phase_shift = np.arctan2(B, A)
    offset_angle = np.arccos(cos_arg)
    
    phi_candidates = [
        phase_shift + offset_angle,
        phase_shift - offset_angle
    ]
    
    # 4. Filter by Aperture
    limit_rad = np.deg2rad(elevation_limit_deg/2.0)
    valid_phis = []
    
    for phi in phi_candidates:
        # Wrap to [-pi, pi]
        phi = np.arctan2(np.sin(phi), np.cos(phi))
        if abs(phi) <= limit_rad:
            valid_phis.append(phi)
            
    if not valid_phis: return None, []

    # Generate 3D points for all valid phis
    candidates = []
    for phi in valid_phis:
        x = r1 * np.cos(theta1) * np.cos(phi)
        y = r1 * np.sin(theta1) * np.cos(phi)
        z = r1 * np.sin(phi)
        candidates.append(np.array([x, y, z]))

    # 5. Determine Worst Case Reprojection Error
    if len(candidates) == 1:
        return candidates[0], candidates
    else:
        errors = []
        for p_cand in candidates:
            # Project to Frame 2
            p2_proj = R @ p_cand + t
            theta2_pred = np.arctan2(p2_proj[1], p2_proj[0])
            err = np.arctan2(np.sin(theta2_pred - theta2), np.cos(theta2_pred - theta2))
            errors.append(abs(err))
        
        worst_idx = np.argmax(errors)
        return candidates[worst_idx], candidates

def calculate_scene(state):
    # -- S1 Geometry (Frame 1 is World for visualization) --
    az_limit1 = np.deg2rad(state['s1_az'])/2.0
    el_limit1 = np.deg2rad(state['s1_el'])/2.0
    # Determine the current range, azimuth, and elevation angles
    r1 = state['s1_min_r'] + (state['s1_max_r'] - state['s1_min_r']) * state['s1_fr']
    theta1 = -az_limit1 + (2 * az_limit1 * state['s1_faz'])
    phi1 = -el_limit1 + (2 * el_limit1 * state['s1_fel'])
    # Determing the Cartesian coordinate, sensor FOV, and ambiguity arc
    p1_local = ut.spherical_to_cartesian(r1, theta1, phi1)
    fov1_local = ut.generate_fov_geometry(state['s1_min_r'], state['s1_max_r'], state['s1_az'], state['s1_el'])
    arc1_local = ut.generate_elevation_arc(r1, theta1, state['s1_el'])

    # -- S2 Geometry (Defined locally, then transformed) --
    az_limit2 = np.deg2rad(state['s2_az'])/2.0
    el_limit2 = np.deg2rad(state['s2_el'])/2.0
    # Determine the current range, azimuth, and elevation angles
    r2 = state['s2_min_r'] + (state['s2_max_r'] - state['s2_min_r']) * state['s2_fr']
    theta2 = -az_limit2 + (2 * az_limit2 * state['s2_faz'])
    phi2 = -el_limit2 + (2 * el_limit2 * state['s2_fel'])
    # Determing the Cartesian coordinate, sensor FOV, and ambiguity arc
    p2_local = ut.spherical_to_cartesian(r2, theta2, phi2)
    fov2_local = ut.generate_fov_geometry(state['s2_min_r'], state['s2_max_r'], state['s2_az'], state['s2_el'])
    arc2_local = ut.generate_elevation_arc(r2, theta2, state['s2_el'])

    # Transformation (Frame 2 -> Frame 1)
    # The slider defines the Rotation of S2 relative to S1
    # We define the translation implicitly by forcing the features in the two sensor frames to overlap
    R_12 = ut.get_rotation_matrix(np.deg2rad(state['roll']), np.deg2rad(state['pitch']), np.deg2rad(state['yaw']))
    s2_origin = p1_local - R_12 @ p2_local
    
    # Transform S2 geometry to Global (Frame 1) for plotting
    fov2_global = [(R_12 @ line.T).T + s2_origin for line in fov2_local]
    arc2_global = (R_12 @ arc2_local.T).T + s2_origin

    # Package measurements for feature estimation
    measurements = {
        'm1': (r1, theta1),
        'm2': (r2, theta2),
        'R_12': R_12,
        's2_origin': s2_origin
    }

    return p1_local, fov1_local, arc1_local, s2_origin, fov2_global, arc2_global, measurements

# --- Global State ---
state = {
    's1_min_r': 1.0, # min range (meters)
    's1_max_r': 10.0, # max range (meters)
    's1_az': 60.0, # azimuth aperture (degrees)
    's1_el': 20.0, # elevation aperture (degrees)
    's1_fr': 0.5, # relative range of feature within sonar min and max (percentage between 0 and 1)
    's1_faz': 0.5, # relative azimoth of feature within sonar FOV (percentage between 0 and 1)
    's1_fel': 0.5, # relative elevation of feature within sonar FOV (percentage between 0 and 1)
    's2_min_r': 1.0, # min range (meters)
    's2_max_r': 10.0, # max range (meters)
    's2_az': 60.0, # azimuth aperture (degrees)
    's2_el': 20.0, # elevation aperture (degrees)
    's2_fr': 0.5, # relative range of feature within sonar min and max (percentage between 0 and 1)
    's2_faz': 0.5, # relative azimoth of feature within sonar FOV (percentage between 0 and 1)
    's2_fel': 0.5, # relative elevation of feature within sonar FOV (percentage between 0 and 1)
    'roll': 0.0, # angle (degrees) between first and second sonar
    'pitch': 45.0, # angle (degrees) between first and second sonar
    'yaw': 0.0, # angle (degrees) between first and second sonar
    'show_est': False # boolean to control whether or not to calculate and show the estimated 3D point
}

# --- Visualization Figure ---
fig_viz = plt.figure(figsize=(10, 8), num="3D Visualization")
ax = fig_viz.add_subplot(111, projection='3d')

def update_plot():
    ax.clear()
    p1, fov1, arc1, s2_org, fov2, arc2, meas = calculate_scene(state)
    
    # Plot first sonar origin, FOV, and elevation arc
    ax.scatter([0], [0], [0], color='black', s=20, label='S1 Origin')
    for line in fov1:
        ax.plot(line[:,0], line[:,1], line[:,2], 'b--', alpha=0.3)
    ax.plot(arc1[:,0], arc1[:,1], arc1[:,2], 'b', linewidth=2)
    # Plot second sonar origin, FOV, and elevation arc
    ax.scatter(s2_org[0], s2_org[1], s2_org[2], color='darkorange', s=20, label='S2 Origin')
    for line in fov2:
        ax.plot(line[:,0], line[:,1], line[:,2], color='orange', linestyle='--', alpha=0.5)
    ax.plot(arc2[:,0], arc2[:,1], arc2[:,2], color='orange', linewidth=2)
    # Plot feature point
    ax.scatter(p1[0], p1[1], p1[2], color='r', s=40, marker='x', label='Ground Truth')
    
    subtitle = ""
    
    # If enabled, solve for the 3D point from the measurements and geometry, and add information to the plot
    if state['show_est']:
        # Prepare transforms for solver: p2 = R_rel * p1 + t_rel
        # Code uses p1 = R_12 * p2 + s2_org
        # Invert: p2 = R_12.T * (p1 - s2_org) = R_12.T * p1 - R_12.T * s2_org
        R_rel = meas['R_12'].T
        t_rel = -meas['R_12'].T @ meas['s2_origin']
        # Determine any possible solutions, and isolate the worst case solution
        worst, candidates = solve_fls_fls(meas['m1'], meas['m2'], R_rel, t_rel, state['s1_el'])
        
        if worst is not None:
            # Calculate Error
            err_dist = np.linalg.norm(worst - p1)
            # Plot Candidates
            if len(candidates) > 1:
                cand_arr = np.array(candidates)
                ax.scatter(cand_arr[:,0], cand_arr[:,1], cand_arr[:,2], color='cyan', s=30, alpha=0.6, label='Ambiguous Sol')
            # Plot Worst Case Estimate
            ax.scatter(worst[0], worst[1], worst[2], color='magenta', s=50, marker='*', label='Estimate')
            # Add Information to Title
            subtitle = f"\nEst: [{worst[0]:.2f}, {worst[1]:.2f}, {worst[2]:.2f}] | Worst Case Error: {err_dist:.4f}m | Sol Found: {len(candidates)}"
        else:
            subtitle = "\nNO SOLUTION FOUND (Check Aperture/Intersection)"

    # Titles and Legends
    title_coords = f"Feature (GT): [{p1[0]:.2f}, {p1[1]:.2f}, {p1[2]:.2f}]"
    title_pose = (f"Pose (S2->S1): R=[{state['roll']:.1f}, {state['pitch']:.1f}, {state['yaw']:.1f}], "
                  f"t=[{s2_org[0]:.2f}, {s2_org[1]:.2f}, {s2_org[2]:.2f}]")
    
    ax.set_title(f"Dual FLS Geometry\n{title_coords}{subtitle}\n{title_pose}", fontsize=10)
    ax.legend(loc='upper right')
    
    # Axis scaling
    all_points = np.vstack([p1, s2_org])
    max_range = max(state['s1_max_r'], state['s2_max_r']) * 1.5
    mid = np.mean(all_points, axis=0)
    ax.set_xlim([mid[0]-max_range/2, mid[0]+max_range/2])
    ax.set_ylim([mid[1]-max_range/2, mid[1]+max_range/2])
    ax.set_zlim([mid[2]-max_range/2, mid[2]+max_range/2])
    ax.set_axis_off()
    fig_viz.canvas.draw_idle()

# --- Controls Figure ---
fig_controls = plt.figure(figsize=(5, 8), num="Controls")
plt.axis('off')

def add_control_row(y_start, label, key_prefix, color_text):
    plt.text(0.05, y_start, label, fontsize=12, weight='bold', color=color_text, transform=fig_controls.transFigure)
    
    # TextBoxes
    y_tb1 = y_start - 0.05
    ax_min = fig_controls.add_axes([0.20, y_tb1, 0.25, 0.03])
    tb_min = TextBox(ax_min, 'MinR ', initial=str(state[f'{key_prefix}_min_r']), label_pad=0.05)
    
    ax_max = fig_controls.add_axes([0.65, y_tb1, 0.25, 0.03])
    tb_max = TextBox(ax_max, 'MaxR ', initial=str(state[f'{key_prefix}_max_r']), label_pad=0.05)
    
    y_tb2 = y_start - 0.1
    ax_az = fig_controls.add_axes([0.20, y_tb2, 0.25, 0.03])
    tb_az = TextBox(ax_az, 'Az° ', initial=str(state[f'{key_prefix}_az']), label_pad=0.05)
    
    ax_el = fig_controls.add_axes([0.65, y_tb2, 0.25, 0.03])
    tb_el = TextBox(ax_el, 'El° ', initial=str(state[f'{key_prefix}_el']), label_pad=0.05)
    
    # Callbacks (simplified for brevity, logic same as before)
    def make_submit(key): return lambda text: update_tb(key, text)
    def update_tb(key, text):
        try: state[key] = float(text); update_plot()
        except: pass
        
    tb_min.on_submit(make_submit(f'{key_prefix}_min_r'))
    tb_max.on_submit(make_submit(f'{key_prefix}_max_r'))
    tb_az.on_submit(make_submit(f'{key_prefix}_az'))
    tb_el.on_submit(make_submit(f'{key_prefix}_el'))

    # Sliders
    y_sl = y_start - 0.15
    step = 0.04
    ax_fr = fig_controls.add_axes([0.25, y_sl, 0.65, 0.025])
    sl_fr = Slider(ax_fr, 'R %', 0.0, 1.0, valinit=state[f'{key_prefix}_fr'])
    ax_faz = fig_controls.add_axes([0.25, y_sl - step, 0.65, 0.025])
    sl_faz = Slider(ax_faz, 'Az %', 0.0, 1.0, valinit=state[f'{key_prefix}_faz'])
    ax_fel = fig_controls.add_axes([0.25, y_sl - 2*step, 0.65, 0.025])
    sl_fel = Slider(ax_fel, 'El %', 0.0, 1.0, valinit=state[f'{key_prefix}_fel'])
    
    return [tb_min, tb_max, tb_az, tb_el], [sl_fr, sl_faz, sl_fel]

tbs1, sls1 = add_control_row(0.95, "Sonar 1 (Blue)", 's1', 'blue')
tbs2, sls2 = add_control_row(0.63, "Sonar 2 (Orange)", 's2', 'darkorange')

# Rotation Controls
y_rot = 0.3
plt.text(0.05, y_rot, "Relative Rotation (S2 to S1)", fontsize=12, weight='bold', transform=fig_controls.transFigure)
ax_roll = fig_controls.add_axes([0.25, y_rot - 0.05, 0.65, 0.025])
sl_roll = Slider(ax_roll, 'Roll', -180, 180, valinit=state['roll'])
ax_pitch = fig_controls.add_axes([0.25, y_rot - 0.09, 0.65, 0.025])
sl_pitch = Slider(ax_pitch, 'Pitch', -180, 180, valinit=state['pitch'])
ax_yaw = fig_controls.add_axes([0.25, y_rot - 0.13, 0.65, 0.025])
sl_yaw = Slider(ax_yaw, 'Yaw', -180, 180, valinit=state['yaw'])

# Checkbox for Estimation
ax_chk = fig_controls.add_axes([0.05, 0.06, 0.4, 0.05])
chk_est = CheckButtons(ax_chk, ['Show Estimation'], [state['show_est']])

def toggle_est(label):
    state['show_est'] = not state['show_est']
    update_plot()
chk_est.on_clicked(toggle_est)

def update_sliders(val):
    state['s1_fr'] = sls1[0].val
    state['s1_faz'] = sls1[1].val
    state['s1_fel'] = sls1[2].val
    state['s2_fr'] = sls2[0].val
    state['s2_faz'] = sls2[1].val
    state['s2_fel'] = sls2[2].val
    state['roll'] = sl_roll.val
    state['pitch'] = sl_pitch.val
    state['yaw'] = sl_yaw.val
    update_plot()

for s in sls1 + sls2 + [sl_roll, sl_pitch, sl_yaw]:
    s.on_changed(update_sliders)

update_plot()
plt.show()