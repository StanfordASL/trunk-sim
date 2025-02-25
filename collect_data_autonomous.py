import mujoco
import mediapy as media
import numpy as np
import pandas as pd
import os

from data_generation_utils import (
    reset_sim,
    apply_random_control, 
    has_converged,
    get_geom_positions,
    apply_random_step,
    get_ee_position
)

# Define the geoms to include
capsule_ids_to_include = [0, 15, 25, 32, 37]  # Selected geoms
geom_names_to_include = [f"actuatedG{i}" for i in capsule_ids_to_include]  # 'actuatedGi' are the geom names

# Define model path and load model and data
current_dir = os.path.dirname(__file__)
xml_path = os.path.join(current_dir, "models", "cable_trunk_expanded_old_4_tendons.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

# Map geom names to their IDs, determined based on Mujoco
geom_ids_to_include = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    for geom_name in geom_names_to_include
]

# Load steady-state positions
folder_path = os.path.join(current_dir, 'trajectories', 'data')
file_path = os.path.join(folder_path, f"steady_state_z_positions.csv")
steady_state_df = pd.read_csv(file_path, header=None)
steady_state_names = steady_state_df.values[0].astype(str)
steady_state_z_values = steady_state_df.values[1:].astype(np.float64).squeeze()

# Simulation parameters
n_trajectories = 300
n_trajectories_step = 50
n_trajectories_random_switch_to_2 = 250
update_frequency = 50 
perturbation_duration = 1.0  # Duration for applying fixed random control input (seconds)
duration = 5.0  # Maximum duration for the simulation following the perturbation (seconds)
framerate = 60  # Simulation framerate (Hz)
control_scale_random = 0.4  # Scale for random control inputs
control_scale_random_2 = 0.2  # Scale for random control inputs
control_scale_step = 0.23  # Scale for random control inputs
window_size = 50  # Number of frames to consider for convergence check
convergence_threshold = 1e-18  # Threshold for considering system at rest
detect_cnvergece = False
create_video = False # bool that decides whether to create a video
include_velocity = True # bool that decides whether to compute the velocity using euler backwards to include it in the observation

# Save initial position and velocity
qpos_init = data.qpos
qpos_init[0] = 0.005855615009311374 # initial index is the slider joint, steady state is not at zero due to the weight, this initializes at steady state
qvel_init = data.qvel

# Simulate, collect state trajectories, and render video
print("Starting simulation...")

for traj in range(n_trajectories):
    # Prepare for collecting data and rendering
    trajectories = []
    frames = []
    converged = False

    # Apply random control input to determine initial conditions
    reset_sim(data, qpos_init, qvel_init)  # Start at steady state

    if traj <= n_trajectories_step:
        apply_random_step(data, model, perturbation_duration, control_scale_step)
    elif traj <= n_trajectories_random_switch_to_2:
        apply_random_control(data, model, perturbation_duration, control_scale_random, update_frequency)
    else:
        apply_random_control(data, model, perturbation_duration, control_scale_random_2, update_frequency)

    # Save the resulting generalized coordinates and velocities
    qpos_init_traj = data.qpos.copy()
    qvel_init_traj = data.qvel.copy()

    # Reset simulation with new initial conditions
    reset_sim(data, qpos_init_traj, qvel_init_traj)        

    # initialize values used for velocity computation
    if include_velocity:
        last_geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
        last_ee_position = get_ee_position(data, ee_site_id, steady_state_z_values)

    with mujoco.Renderer(model) as renderer:
        while data.time < duration and not converged:
            mujoco.mj_step(model, data)

            # Record positions of the selected geoms
            geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
            ee_delta_xpos = get_ee_position(data, ee_site_id, steady_state_z_values)
            
            # Record positions of the selected geoms
            if include_velocity:
                geom_velocities = []
                for last_velocity_idx in range(len(last_geom_positions)):
                    geom_velocities.append((geom_positions[last_velocity_idx] - last_geom_positions[last_velocity_idx])/model.opt.timestep)  # Append x, y, z, velcities 
                last_geom_positions = geom_positions

                # ee
                ee_velocity = ee_delta_xpos - last_ee_position
                last_ee_position = ee_delta_xpos

            # Combine time, geom positions, and end-effector data into state
            state = [data.time] + ee_delta_xpos.tolist() + np.hstack(geom_positions).tolist()
            if include_velocity:
                state += ee_velocity.tolist()
                state += np.hstack(geom_velocities).tolist()

            trajectories.append(state)

            # Render the frame
            if create_video and len(frames) < data.time * framerate:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

            # Check for convergence
            recent_states = [state[1:] for state in trajectories]  # Skip time column
            if detect_cnvergece:
                converged = has_converged(recent_states, convergence_threshold, window_size)
    
    # Define the subfolder paths
    data_folder = os.path.join(current_dir, "trajectories", "data_lukas_more")
    video_folder = os.path.join(current_dir, "trajectories", "videos_lukas_more")

    # Generate dynamic column names
    geom_columns = [
        f"{axis}_{geom_names_to_include[i]}"
        for i in range(len(geom_names_to_include))
        for axis in ['x', 'y', 'z']
    ]
    columns = ["time"] + ["x_ee", "y_ee", "z_ee"] + geom_columns
    if include_velocity:
        geom_velocity_columns = [
        f"{axis}_velocity_{geom_names_to_include[i]}"
        for i in range(len(geom_names_to_include))
        for axis in ['x', 'y', 'z']
        ]
        columns += ["x_velocity_ee", "y_velocity_ee", "z_velocity_ee"] + geom_velocity_columns

    # Save trajectories to a CSV file
    df = pd.DataFrame(trajectories, columns=columns)
    output_csv_file = os.path.join(data_folder, f"autonomous_state_traj_{traj}.csv")
    df.to_csv(output_csv_file, index=False)
    print(f"State trajectory {traj} saved to {output_csv_file}.")

    # Save rendered video
    if create_video:
        output_video_file = os.path.join(video_folder, f"video_autonomous_state_traj_{traj}.mp4")
        media.write_video(output_video_file, frames, fps=framerate)
        print(f"Simulation video saved to {output_video_file}.")

print("Simulation finished.")

