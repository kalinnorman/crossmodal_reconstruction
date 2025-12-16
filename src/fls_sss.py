import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D

import utils as ut

def calculate_scene(state):
    # FLS Geometry
    p1_local = ut.spherical_to_cartesian(
        state['s1_min_r'] + (state['s1_max_r'] - state['s1_min_r']) * state['s1_fr'],
        np.deg2rad(state['s1_az'])/2.0 * (2*state['s1_faz'] - 1),
        np.deg2rad(state['s1_el'])/2.0 * (2*state['s1_fel'] - 1)
    )
    
    fov1_local = ut.generate_fov_geometry(state['s1_min_r'], state['s1_max_r'], state['s1_az'], state['s1_el'])
    
    # FLS Feature 3D Ambiguity - we know range and azimuth, but unknown elevation
    az_limit1 = np.deg2rad(state['s1_az'])/2.0
    theta1 = -az_limit1 + (2 * az_limit1 * state['s1_faz'])
    r1 = np.linalg.norm(p1_local)
    arc1_local = ut.generate_elevation_arc(r1, theta1, state['s1_el'])

    # SSS Geometry
    p2_local = ut.spherical_to_cartesian(
        state['s2_min_r'] + (state['s2_max_r'] - state['s2_min_r']) * state['s2_fr'],
        np.deg2rad(state['s2_az'])/2.0 * (2*state['s2_faz'] - 1),
        np.deg2rad(state['s2_el'])/2.0 * (2*state['s2_fel'] - 1)
    )
    
    fov2_local = ut.generate_fov_geometry(state['s2_min_r'], state['s2_max_r'], state['s2_az'], state['s2_el'])
    
    # SSS Feature 3D Ambiguity - we know range, but unknown azimuth and elevation
    r2 = np.linalg.norm(p2_local)
    sss_surface_local = ut.generate_azimuth_elevation_surface_lines(r2, state['s2_az'], state['s2_el'], density=8)

    # Transformation
    R = ut.get_rotation_matrix(np.deg2rad(state['roll']), np.deg2rad(state['pitch']), np.deg2rad(state['yaw']))
    s2_origin = p1_local - R @ p2_local
    
    # Transform FOV lines
    fov2_global = []
    for line in fov2_local:
        line_global = (R @ line.T).T + s2_origin
        fov2_global.append(line_global)
        
    # Transform ambiguity surface lines
    sss_surface_global = []
    for line in sss_surface_local:
        line_global = (R @ line.T).T + s2_origin
        sss_surface_global.append(line_global)

    return p1_local, fov1_local, arc1_local, s2_origin, fov2_global, sss_surface_global

# --- Global State ---
state = {
    # FLS Params
    's1_min_r': 1.0, # min range (meters)
    's1_max_r': 10.0, # max range (meters)
    's1_az': 60.0, # azimuth aperture (degrees)
    's1_el': 20.0, # elevation aperture (degrees)
    's1_fr': 0.5, # relative range of feature within sonar min and max (percentage between 0 and 1)
    's1_faz': 0.5, # relative azimoth of feature within sonar FOV (percentage between 0 and 1)
    's1_fel': 0.5, # relative elevation of feature within sonar FOV (percentage between 0 and 1)
    # SSS Params 
    's2_min_r': 1.0, # min range (meters)
    's2_max_r': 15.0, # max range (meters)
    's2_az': 120.0, # azimuth aperture (degrees)
    's2_el': 3.0, # elevation aperture (degrees)
    's2_fr': 0.5, # relative range of feature within sonar min and max (percentage between 0 and 1)
    's2_faz': 0.5, # relative azimoth of feature within sonar FOV (percentage between 0 and 1)
    's2_fel': 0.5, # relative elevation of feature within sonar FOV (percentage between 0 and 1)
    # Pose
    'roll': 0.0, # angle (degrees) between first and second sonar
    'pitch': 90.0, # angle (degrees) between first and second sonar
    'yaw': 0.0 # angle (degrees) between first and second sonar
}

# Figure 1: 3D visualization of sonar FOVs and feature location and ambiguity
fig_viz = plt.figure(figsize=(10, 8), num="FLS & SSS Visualization")
ax = fig_viz.add_subplot(111, projection='3d')

def update_plot():
    ax.clear()
    p1, fov1, arc1, s2_org, fov2, sss_surface = calculate_scene(state)
    # Plot FLS (Blue)
    ax.scatter([0], [0], [0], color='black', s=20, label='FLS Origin')
    for line in fov1:
        ax.plot(line[:,0], line[:,1], line[:,2], 'b--', alpha=0.3)
    ax.plot(arc1[:,0], arc1[:,1], arc1[:,2], 'b', linewidth=2)
    # Plot SSS (Orange)
    ax.scatter(s2_org[0], s2_org[1], s2_org[2], color='darkorange', s=20, label='SSS Origin')
    for line in fov2:
        ax.plot(line[:,0], line[:,1], line[:,2], color='orange', linestyle='--', alpha=0.3)
    # Plot SSS Ambiguity Surface (Grid)
    # We iterate through the lines that make up the surface
    for i, line in enumerate(sss_surface):
        # Only label the first line to avoid cluttering legend
        ax.plot(line[:,0], line[:,1], line[:,2], color='orange', alpha=0.6, linewidth=1)
    # Plot Feature
    ax.scatter(p1[0], p1[1], p1[2], color='r', s=40, label='Feature')
    
    # Set up subtitles that will display the feature location, and relative pose values
    title_coords = f"Feature in FLS Frame: x={p1[0]:.2f}, y={p1[1]:.2f}, z={p1[2]:.2f}"
    title_pose = (f"Rotation (deg): R={state['roll']:.1f}, P={state['pitch']:.1f}, Y={state['yaw']:.1f}\n"
                  f"Translation: tx={s2_org[0]:.2f}, ty={s2_org[1]:.2f}, tz={s2_org[2]:.2f}")
    # Add the title and legend
    ax.set_title(f"Cross-Modal Geometry (FLS to SSS)\n{title_coords}\n{title_pose}", fontsize=10)
    ax.legend(loc='upper right')
    # Set axis limits based of off points currently on the plot
    all_points = np.vstack([p1, s2_org])
    max_range = max(state['s1_max_r'], state['s2_max_r']) * 1.5
    mid = np.mean(all_points, axis=0)
    ax.set_xlim([mid[0]-max_range/2, mid[0]+max_range/2])
    ax.set_ylim([mid[1]-max_range/2, mid[1]+max_range/2])
    ax.set_zlim([mid[2]-max_range/2, mid[2]+max_range/2])
    # Turn off the axis to reduce visual clutter
    ax.set_axis_off()
    # Update the figure
    fig_viz.canvas.draw_idle()

# Figure 2: Parameter controls 
fig_controls = plt.figure(figsize=(5, 10), num="Controls")
plt.axis('off')

# Logic for TextBoxes
def add_control_row(y_start, label, key_prefix, color_text):
    plt.text(0.05, y_start, label, fontsize=12, weight='bold', color=color_text, transform=fig_controls.transFigure)
    
    # --- Min Range ---
    y_tb1 = y_start - 0.04
    ax_min = fig_controls.add_axes([0.20, y_tb1, 0.25, 0.03])
    tb_min = TextBox(ax_min, 'MinR ', initial=str(state[f'{key_prefix}_min_r']), label_pad=0.05)
    
    def submit_min(text):
        key = f'{key_prefix}_min_r'
        key_max = f'{key_prefix}_max_r'
        try:
            val = float(text)
            if 0 < val < state[key_max]:
                state[key] = val
                update_plot()
        except ValueError: pass
        tb_min.set_val(str(state[key]))
        
    tb_min.on_submit(submit_min)
    
    # --- Max Range ---
    ax_max = fig_controls.add_axes([0.65, y_tb1, 0.25, 0.03])
    tb_max = TextBox(ax_max, 'MaxR ', initial=str(state[f'{key_prefix}_max_r']), label_pad=0.05)
    
    def submit_max(text):
        key = f'{key_prefix}_max_r'
        key_min = f'{key_prefix}_min_r'
        try:
            val = float(text)
            if val > state[key_min]:
                state[key] = val
                update_plot()
        except ValueError: pass
        tb_max.set_val(str(state[key]))

    tb_max.on_submit(submit_max)
    
    # --- Azimuth Aperture ---
    y_tb2 = y_start - 0.08
    ax_az = fig_controls.add_axes([0.20, y_tb2, 0.25, 0.03])
    tb_az = TextBox(ax_az, 'Az° ', initial=str(state[f'{key_prefix}_az']), label_pad=0.05)
    
    def submit_az(text):
        key = f'{key_prefix}_az'
        try:
            val = float(text)
            if 0 < val <= 360:
                state[key] = val
                update_plot()
        except ValueError: pass
        tb_az.set_val(str(state[key]))
        
    tb_az.on_submit(submit_az)
    
    # --- Elevation Aperture ---
    ax_el = fig_controls.add_axes([0.65, y_tb2, 0.25, 0.03])
    tb_el = TextBox(ax_el, 'El° ', initial=str(state[f'{key_prefix}_el']), label_pad=0.05)
    
    def submit_el(text):
        key = f'{key_prefix}_el'
        try:
            val = float(text)
            if 0 < val <= 180:
                state[key] = val
                update_plot()
        except ValueError: pass
        tb_el.set_val(str(state[key]))
        
    tb_el.on_submit(submit_el)
    
    # --- Sliders for Feature Location ---
    y_sl = y_start - 0.13
    step = 0.04
    
    ax_fr = fig_controls.add_axes([0.25, y_sl, 0.65, 0.025])
    sl_fr = Slider(ax_fr, 'R %', 0.0, 1.0, valinit=state[f'{key_prefix}_fr'])
    
    ax_faz = fig_controls.add_axes([0.25, y_sl - step, 0.65, 0.025])
    sl_faz = Slider(ax_faz, 'Az %', 0.0, 1.0, valinit=state[f'{key_prefix}_faz'])
    
    ax_fel = fig_controls.add_axes([0.25, y_sl - 2*step, 0.65, 0.025])
    sl_fel = Slider(ax_fel, 'El %', 0.0, 1.0, valinit=state[f'{key_prefix}_fel'])
    
    return [tb_min, tb_max, tb_az, tb_el], [sl_fr, sl_faz, sl_fel]

# Add FLS controls
tbs1, sls1 = add_control_row(0.92, "FLS (Blue)", 's1', 'blue')

# Add SSS controls
tbs2, sls2 = add_control_row(0.60, "Sidescan (Orange)", 's2', 'darkorange')

# Add rotation controls
y_rot = 0.25
plt.text(0.05, y_rot, "Relative Rotation (SSS to FLS)", fontsize=12, weight='bold', transform=fig_controls.transFigure)

ax_roll = fig_controls.add_axes([0.25, y_rot - 0.05, 0.65, 0.025])
sl_roll = Slider(ax_roll, 'Roll', -180, 180, valinit=state['roll'])

ax_pitch = fig_controls.add_axes([0.25, y_rot - 0.09, 0.65, 0.025])
sl_pitch = Slider(ax_pitch, 'Pitch', -180, 180, valinit=state['pitch'])

ax_yaw = fig_controls.add_axes([0.25, y_rot - 0.13, 0.65, 0.025])
sl_yaw = Slider(ax_yaw, 'Yaw', -180, 180, valinit=state['yaw'])

# Sliders update function
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

# Connect all sliders to the update function
for s in sls1 + sls2 + [sl_roll, sl_pitch, sl_yaw]:
    s.on_changed(update_sliders)

# IMPORTANT: Keep references to ALL widgets (text boxes AND sliders), otherwise the text boxes will be garbage collected and become unresponsive
_widgets = [tbs1, sls1, tbs2, sls2, sl_roll, sl_pitch, sl_yaw]

update_plot()
plt.show()