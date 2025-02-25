import mujoco
import mediapy as media
import numpy as np
import pandas as pd
import os

from data_generation_utils import (
    reset_sim,
    get_geom_positions,
    get_ee_position,
    compute_u_circle_sym
)

# Load model and data
current_dir = os.path.dirname(__file__)
xml_path = os.path.abspath(os.path.join(current_dir, '..', "models", "cable_trunk_expanded_old_4_tendons.xml"))
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Define hyperparameters
duration = 15.0  # Maximum simulation duration (seconds)
framerate = 60  # Simulation framerate (Hz)
control_scale = 0.1  # Original control scale
create_video = True  # Whether to create a video
include_velocity = True  # Whether to compute and save velocities

# ---- Experiment-specific parameters for variable angular velocity ----
# Fix the circle radius to be half of the original maximum amplitude.
fixed_amplitude = control_scale

# Define the range for the circle’s angular velocity (frequency in rad/s).
min_circle_frequency = 0.05  # Minimum angular velocity
max_circle_frequency = 1.5  # Maximum angular velocity

step_input_range = 0.008
num_trajectories = 30  # Number of trajectories to generate

# Output folder setup (separate folder for the variable angular velocity experiment)
folder_path = os.path.abspath(os.path.join(current_dir, '..', 'trajectories_b_test', 'data_controlled_faster_larger'))
os.makedirs(folder_path, exist_ok=True)
existing_files = [f for f in os.listdir(folder_path) if f.startswith("controlled_trajectory") and f.endswith(".csv")]
trajectory_start_index = len(existing_files)

# Steady state data (assumed to be in the parent folder)
file_path = os.path.abspath(os.path.join(folder_path, '..', f"steady_state_z_positions.csv"))
steady_state_df = pd.read_csv(file_path, header=None)
steady_state_names = steady_state_df.values[0].astype(str)
steady_state_z_values = steady_state_df.values[1:].astype(np.float64).squeeze()

# Save initial position and velocity
qpos_init = data.qpos.copy()
qpos_init[0] = 0.005855615009311374  # Initial slider joint position at steady state
qvel_init = data.qvel.copy()

# End-effector (ee) site ID
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

# (Optional) Steady state positions used for calibration (if needed)
steady_state_position_slow = np.array([
    5.00200000e+00, 3.19090704e-07, -1.68952692e-07, 5.85561501e-03,
    1.12948442e-30, 0.00000000e+00, 5.85561501e-03, 6.93124410e-08,
    -3.67153785e-08, 5.85561501e-03, 1.62986945e-07, -8.63127160e-08,
    5.85561501e-03, 2.39974259e-07, -1.27072120e-07, 5.85561501e-03,
    2.97070643e-07, -1.57296843e-07, 5.85561501e-03, -6.33629856e-09,
    3.11889406e-09, -6.07847106e-15, 2.44593395e-29, 0.00000000e+00,
    -1.11022302e-13, -6.93258778e-07, 3.41355222e-07, -4.44089210e-13,
    -1.62280607e-06, 7.98891345e-07, -1.30451205e-12, -2.38590475e-06,
    1.17447969e-06, -2.10942375e-12, -2.95060859e-06, 1.45239239e-06,
    -2.78943535e-12, -4.44373085e-05, -2.84491509e-05, -6.08491182e-05,
    -3.30466712e-05, -4.44373085e-05, -2.84491509e-05, -6.08491182e-05,
    -3.30466712e-05
])
steady_state_position_slow_osc = np.array([
    5.00200000e+00, -1.40185460e-06, 8.25550509e-07, 5.85561501e-03,
    1.33156284e-29, 0.00000000e+00, 5.85561501e-03, -3.04474394e-07,
    1.80313492e-07, 5.85561501e-03, -7.16016170e-07, 4.22567273e-07,
    5.85561501e-03, -1.05425168e-06, 6.21498711e-07, 5.85561501e-03,
    -1.30510692e-06, 7.68792369e-07, 5.85561501e-03, 2.83559141e-08,
    -1.53790648e-09, -9.98923166e-14, 4.14776322e-28, 0.00000000e+00,
    -1.88737914e-12, 3.10218759e-06, -1.75492705e-07, -8.21565038e-12,
    7.26208092e-06, -4.00370788e-07, -2.25652830e-11, 1.06771291e-05,
    -5.83757693e-07, -3.59157148e-11, 1.32043747e-05, -7.17704177e-07,
    -4.60464999e-11, 0, 0, 0, 0, 0, 0, 0, 0
])

# Generate multiple trajectories with increasing angular velocity
for traj_idx in range(num_trajectories):
    print(f"Starting controlled simulation for trajectory {traj_idx + 1} with variable angular velocity...")
    reset_sim(data, qpos_init, qvel_init)  # Reinitialize simulation at steady state

    trajectory = []
    frames = []

    # Number of control inputs used in the model (assumed to be half of model.nu)
    num_controls = model.nu // 2
    offsets = np.full(num_controls, control_scale / 2)

    # Use fixed amplitude (circle radius) for all trajectories.
    amplitudes_circle = fixed_amplitude

    # Compute the angular velocity for this trajectory (linearly spaced).
    frequency = min_circle_frequency + (traj_idx / (num_trajectories - 1)) * (
                max_circle_frequency - min_circle_frequency)

    # Sample step control values (if used in your step logic)
    step_control_values = np.random.uniform(-step_input_range, step_input_range, num_controls)
    step_moment = 3  # timepoint at which a step change is introduced

    print("Defining lambda matrix for the desired dynamics.")
    lambda_matrix = np.array([
        [-2., 0.],
        [0., -2.5],
    ])

    # Identify the capsule/geoms to be tracked
    capsule_ids_to_include = [0, 15, 25, 32, 37]
    geom_ids_to_include = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"actuatedG{i}")
        for i in capsule_ids_to_include
    ]
    geom_names_to_include = [f"actuatedG{i}" for i in capsule_ids_to_include]

    # Initialize variables for velocity computation if required
    if include_velocity:
        last_geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
        last_ee_position = get_ee_position(data, ee_site_id, steady_state_z_values)

    # Initialize control variable (assumed to be half the length of the full control vector)
    current_real_control = np.zeros(num_controls // 2)

    with mujoco.Renderer(model) as renderer:
        last_time = data.time
        while data.time < duration:
            time = data.time
            last_real_control = current_real_control
            # Compute control inputs for the circle using the fixed amplitude and variable angular velocity.
            current_real_control, current_desired_control = compute_u_circle_sym(
                time, amplitudes_circle, frequency, lambda_matrix,
                num_controls, last_time, last_real_control, step_moment
            )


            def expand_control(u):
                # Expand from half the control dimension to full control space (example: symmetric actuation)
                return np.array([u[0], -u[0], u[1], -u[1]])


            current_real_control_full = expand_control(current_real_control)
            data.ctrl[:] = current_real_control_full
            last_time = time

            mujoco.mj_step(model, data)

            # Record positions of the selected geoms and the end-effector
            geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
            ee_delta_xpos = get_ee_position(data, ee_site_id, steady_state_z_values)

            if include_velocity:
                geom_velocities = []
                for idx in range(len(last_geom_positions)):
                    geom_velocities.append(
                        (geom_positions[idx] - last_geom_positions[idx]) / model.opt.timestep
                    )
                last_geom_positions = geom_positions
                ee_velocity = ee_delta_xpos - last_ee_position
                last_ee_position = ee_delta_xpos

            # Build state: time, ee positions, geom positions, (velocities if enabled), and control inputs.
            state = [data.time] + ee_delta_xpos.tolist() + np.hstack(geom_positions).tolist()
            if include_velocity:
                state += ee_velocity.tolist()
                state += np.hstack(geom_velocities).tolist()
            state += current_real_control.tolist() + current_desired_control.tolist()
            trajectory.append(state)

            # Render frame for video creation
            if create_video and len(frames) < data.time * framerate:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

        # Compute steady-state offset using the mean of the last 10 points before the step input
        step_index = next((i for i, row in enumerate(trajectory) if row[0] >= step_moment), len(trajectory))
        offset_start_idx = max(0, step_index - 10)
        steady_state_offset = np.mean(trajectory[offset_start_idx:step_index], axis=0)
        print(f"Steady-state reference for trajectory {traj_idx + 1}: {steady_state_offset}")

        # Calibrate the first 18 state variables using the steady-state offset
        for i in range(len(trajectory)):
            trajectory[i][1:19] = np.array(trajectory[i][1:19]) - steady_state_offset[1:19]

    # Define column names for the CSV file
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
    control_columns = [f"control_{i}" for i in range(model.nu // 2)]
    control_columns_desired = [f"control_desired_{i}" for i in range(model.nu // 2)]
    columns = ["time"] + ["x_ee", "y_ee", "z_ee"] + geom_columns
    if include_velocity:
        columns += ["x_velocity_ee", "y_velocity_ee", "z_velocity_ee"] + geom_velocity_columns
    columns += control_columns + control_columns_desired
    df = pd.DataFrame(trajectory, columns=columns)

    # Save trajectory CSV
    output_csv_file = os.path.join(folder_path, f"controlled_trajectory_{trajectory_start_index + traj_idx}.csv")
    df.to_csv(output_csv_file, index=False)
    print(f"Controlled trajectory {traj_idx + 1} saved to {output_csv_file}.")

    # Save rendered video (if enabled)
    if create_video:
        video_folder = os.path.abspath(
            os.path.join(current_dir, '..', 'trajectories_b_test', 'videos_controlled_faster_larger'))
        os.makedirs(video_folder, exist_ok=True)
        output_video_file = os.path.join(video_folder,
                                         f"controlled_trajectory_video_{trajectory_start_index + traj_idx}.mp4")
        media.write_video(output_video_file, frames, fps=framerate)
        print(f"Simulation video for trajectory {traj_idx + 1} saved to {output_video_file}.")

print("Controlled simulations with variable angular velocity finished.")
