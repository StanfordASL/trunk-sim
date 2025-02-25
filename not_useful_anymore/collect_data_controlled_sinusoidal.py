import mujoco
import mediapy as media
import numpy as np
import pandas as pd
import os

from data_generation_utils import (
    reset_sim,
    get_geom_positions,
    get_ee_position
)

# Load model and data
current_dir = os.path.dirname(__file__)
xml_path = os.path.join(current_dir, "models", "cable_trunk_expanded_old_4_tendons.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Define hyperparameters
duration = 5.0  # Maximum duration for the simulation (seconds)
framerate = 60  # Simulation framerate (Hz)
control_scale = 0.15  # Scale for control inputs
create_video = True  # Whether to create a video
include_velocity = True  # Whether to compute and save velocities
w_max = 2 * np.pi * 0.1 # Maximum frequency for sinusoidal control
w_min = 2 * np.pi * 0.05 # Minimum frequency for sinusoidal control
num_trajectories = 100  # Number of trajectories to generate

# Output folder setup
folder_path = os.path.join(current_dir, 'trajectories', 'data')
os.makedirs(folder_path, exist_ok=True)
existing_files = [f for f in os.listdir(folder_path) if f.startswith("controlled_trajectory") and f.endswith(".csv")]
trajectory_start_index = len(existing_files)

# Steady state data
file_path = os.path.join(folder_path, f"steady_state_z_positions.csv")
steady_state_df = pd.read_csv(file_path, header=None)
steady_state_names = steady_state_df.values[0].astype(str)
steady_state_z_values = steady_state_df.values[1:].astype(np.float64).squeeze()

# Save initial position and velocity
qpos_init = data.qpos
qpos_init[0] = 0.005855615009311374  # Initial slider joint position at steady state
qvel_init = data.qvel

# ee Id
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

# Generate multiple trajectories
for traj_idx in range(num_trajectories):
    print(f"Starting controlled simulation for trajectory {traj_idx + 1}...")
    reset_sim(data, qpos_init, qvel_init)  # Initialize at steady state

    trajectory = []
    frames = []

    # Sample sinusoidal coefficients for each control input
    num_controls = model.nu
    print("Number of control inputs used: ", num_controls)
    offsets = np.full(num_controls, control_scale / 2)
    amplitudes = np.random.uniform(0, control_scale / 2, num_controls)
    frequencies = np.random.uniform(w_min, w_max, num_controls)
    
    capsule_ids_to_include = [0, 15, 25, 32, 37]
    geom_ids_to_include = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"actuatedG{i}")
        for i in capsule_ids_to_include
    ]
    geom_names_to_include = [f"actuatedG{i}" for i in capsule_ids_to_include]
    
    # Initialize variables for velocity computation
    if include_velocity:
        last_geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
        last_ee_position = get_ee_position(data, ee_site_id, steady_state_z_values)

    with mujoco.Renderer(model) as renderer:
        while data.time < duration:
            # Compute control input based on sinusoidal coefficients
            time = data.time
            current_control = offsets + amplitudes * np.sin(frequencies * time)
            data.ctrl[:] = current_control

            mujoco.mj_step(model, data)

            # Record positions of the selected geoms
            geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
            ee_delta_xpos = get_ee_position(data, ee_site_id, steady_state_z_values)

            if include_velocity:
                geom_velocities = []
                for last_velocity_idx in range(len(last_geom_positions)):
                    geom_velocities.append(
                        (geom_positions[last_velocity_idx] - last_geom_positions[last_velocity_idx]) / model.opt.timestep
                    )
                last_geom_positions = geom_positions

                # ee
                ee_velocity = ee_delta_xpos - last_ee_position
                last_ee_position = ee_delta_xpos

            # Combine time, geom positions, end-effector data, velocities (if included), and control inputs into state
            state = [data.time] + ee_delta_xpos.tolist() + np.hstack(geom_positions).tolist()
            if include_velocity:
                state += ee_velocity.tolist()
                state += np.hstack(geom_velocities).tolist()
            state += current_control.tolist()
            trajectory.append(state)

            # Render the frame
            if create_video and len(frames) < data.time * framerate:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

    # Save trajectories to a CSV file
    geom_columns = [
        f"{axis}_{geom_names_to_include[i]}"
        for i in range(len(geom_names_to_include))
        for axis in ['x', 'y', 'z']
    ]
    if include_velocity:
        geom_velocity_columns = [
            f"{axis}_velocity_{geom_names_to_include[i]}"
            for i in range(len(geom_names_to_include))
            for axis in ['x', 'y', 'z']
        ]
    control_columns = [f"control_{i}" for i in range(model.nu)]
    columns = ["time"] + ["x_ee", "y_ee", "z_ee"] + geom_columns
    if include_velocity:
        columns += ["x_velocity_ee", "y_velocity_ee", "z_velocity_ee"] + geom_velocity_columns
    columns += control_columns
    df = pd.DataFrame(trajectory, columns=columns)

    output_csv_file = os.path.join(folder_path, f"controlled_trajectory_{trajectory_start_index + traj_idx}.csv")
    df.to_csv(output_csv_file, index=False)
    print(f"Controlled trajectory {traj_idx + 1} saved to {output_csv_file}.")

    # Save rendered video
    if create_video:
        video_folder = os.path.join(current_dir, "trajectories", "videos")
        os.makedirs(video_folder, exist_ok=True)
        output_video_file = os.path.join(video_folder, f"controlled_trajectory_video_{trajectory_start_index + traj_idx}.mp4")
        media.write_video(output_video_file, frames, fps=framerate)
        print(f"Simulation video for trajectory {traj_idx + 1} saved to {output_video_file}.")

print("Controlled simulations finished.")
