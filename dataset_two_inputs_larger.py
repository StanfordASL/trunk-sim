import os
import numpy as np
import pandas as pd
import mujoco
import mediapy as media

from data_generation_utils import (
    reset_sim,
    has_converged,
    get_geom_positions,
    get_ee_position,
    get_new_lambda_u,
    generate_matrix_with_complex_eigenvalues,
    apply_random_control_symmetric_update  # Using the updated function that returns (setpoint, real)
)

# =============================================================================
# Helper: Generate initial conditions and the initial control signal for actuator dynamics.
# =============================================================================
def generate_initial_conditions(n_trajectories, model, data, qpos_init, qvel_init,
                                update_frequency, perturbation_duration, update_apply_duration,
                                control_scale_random, update_limit,
                                initial_mean, initial_noise_range, lambda_matrix):

    initial_conditions = []  # Each element is a (qpos, qvel) tuple.
    initial_controls = []    # For the actuator branch.
    for traj in range(n_trajectories):
        reset_sim(data, qpos_init, qvel_init)
        # Call the updated function. It now returns a tuple:
        # (last_applied_control_setpoint, last_applied_control_real)
        u_setpoint, u_real = apply_random_control_symmetric_update(
            data, model, perturbation_duration, update_apply_duration,
            control_scale_random, update_limit,
            initial_mean, initial_noise_range, lambda_matrix
        )
        initial_conditions.append((data.qpos.copy(), data.qvel.copy()))
        # Use the real control as the u_init for simulation.
        initial_controls.append(u_real)
    return initial_conditions, initial_controls

# =============================================================================
# Simulation with actuator dynamics (i.e. injecting a computed control at every step)
# =============================================================================
def simulate_actuator(initial_condition, u_init, model, data, duration,
                      include_velocity, geom_ids_to_include, steady_state_z_values,
                      ee_site_id, framerate, create_video, lambda_matrix,
                      detect_cnvergece, convergence_threshold, window_size):
    qpos_ic, qvel_ic = initial_condition
    reset_sim(data, qpos_ic, qvel_ic)

    trajectories = []
    frames = []
    converged = False

    if include_velocity:
        last_geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
        last_ee_position = get_ee_position(data, ee_site_id, steady_state_z_values)

    while data.time < duration and not converged:
        # --- Compute and Apply the Current Control ---
        # u_init is a 2-element vector (the independent control inputs).
        raw_control = get_new_lambda_u(data.time, lambda_matrix, u_init)
        # Expand to 4 channels: [raw_control[0], -raw_control[0], raw_control[1], -raw_control[1]]
        current_control = np.array([raw_control[0], -raw_control[0], raw_control[1], -raw_control[1]])
        data.ctrl[:] = current_control

        mujoco.mj_step(model, data)
        geom_positions = get_geom_positions(geom_ids_to_include, data, steady_state_z_values)
        ee_delta_xpos = get_ee_position(data, ee_site_id, steady_state_z_values)

        if include_velocity:
            geom_velocities = []
            for i in range(len(last_geom_positions)):
                vel = (np.array(geom_positions[i]) - np.array(last_geom_positions[i])) / model.opt.timestep
                geom_velocities.append(vel)
            ee_velocity = (np.array(ee_delta_xpos) - np.array(last_ee_position)) / model.opt.timestep
            last_geom_positions = geom_positions
            last_ee_position = ee_delta_xpos

        state = [data.time] + ee_delta_xpos.tolist() + np.hstack(geom_positions).tolist()
        if include_velocity:
            state += ee_velocity.tolist() + np.hstack(geom_velocities).tolist()
        state += raw_control.tolist()  # Append the independent control values.
        trajectories.append(state)

        if create_video and len(frames) < data.time * framerate:
            with mujoco.Renderer(model) as renderer:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

        if detect_cnvergece:
            recent_states = [s[1:] for s in trajectories]
            converged = has_converged(recent_states, convergence_threshold, window_size)

    return trajectories, frames

# =============================================================================
# Helper to get the next available trajectory index based on existing files.
# =============================================================================
def get_max_index(directory, prefix="state_traj_", suffix=".csv"):
    indices = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(suffix):
            try:
                index = int(file[len(prefix):-len(suffix)])
                indices.append(index)
            except ValueError:
                continue
    return max(indices) if indices else -1

# =============================================================================
# Main Script
# =============================================================================
def main():
    # --- Model and Data Setup ---
    current_dir = os.path.dirname(__file__)
    xml_path = os.path.abspath(os.path.join(current_dir, "..", "models", "cable_trunk_expanded_old_4_tendons.xml"))
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

    # Define the geoms of interest.
    capsule_ids_to_include = [0, 15, 25, 32, 37]
    geom_names_to_include = [f"actuatedG{i}" for i in capsule_ids_to_include]
    geom_ids_to_include = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in geom_names_to_include]

    # Load steady-state positions to “center” the recorded geom positions.
    steady_state_csv_path = os.path.abspath(os.path.join(current_dir, "..", "trajectories_b_test", "steady_state_z_positions.csv"))
    steady_state_df = pd.read_csv(steady_state_csv_path, header=None)
    steady_state_names = steady_state_df.values[0].astype(str)
    steady_state_z_values = steady_state_df.values[1:].astype(np.float64).squeeze()

    # --- Simulation Parameters ---
    n_trajectories = 120
    update_frequency = 50
    perturbation_duration = 0.6  # was 0.5      # seconds for the random control phase
    update_apply_duration = 0.3  # was 0.2    # Duration during which the update is applied (e.g., last 0.1 s)
    duration = 6.0                   # seconds for the simulation after initial perturbation
    framerate = 60                   # Hz (for video recording)
    control_scale_random = 1.0       # Maximum allowed absolute value for control inputs.
    update_limit = np.pi // 2        # Maximum allowed update added at each control update.
    include_velocity = True          # Whether to compute velocities
    create_video = True              # Set to True to record videos

    # New parameters for initial control generation.
    initial_mean = 0.9               # Baseline magnitude (will be ±0.4)
    initial_noise_range = 0.2        # Uniform noise in [-0.1, 0.1]

    # Convergence-check parameters.
    detect_cnvergece = False         # Set True to halt simulation when convergence is detected.
    convergence_threshold = 1e-18
    window_size = 50

    # --- Actuator Dynamics Setup ---
    # Use two eigenvalues (model uses only two independent inputs).
    eigenvalues = [-2, -2.5]
    lambda_matrix = generate_matrix_with_complex_eigenvalues(eigenvalues)
    print("This is the lambda matrix being used for simulation:", lambda_matrix)

    # Set the steady-state base (slider joint initialization as in your scripts).
    qpos_init = data.qpos.copy()
    qpos_init[0] = 0.005855615009311374
    qvel_init = data.qvel.copy()

    # --- Generate Initial Conditions ---
    print("Generating initial conditions for all trajectories...")
    # Note: We now pass lambda_matrix so that the updated function returns the real control.
    initial_conditions, initial_controls = generate_initial_conditions(
        n_trajectories, model, data, qpos_init, qvel_init,
        update_frequency, perturbation_duration, update_apply_duration,
        control_scale_random, update_limit,
        initial_mean, initial_noise_range, lambda_matrix
    )
    print("Initial conditions generated.")

    # --- Setup Output Directories for Actuator Dynamics (with lambda) ---
    data_folder_actuator = os.path.abspath(os.path.join(current_dir, "..", "larger_sets", "data_even_larger2"))
    video_folder_actuator = os.path.abspath(os.path.join(current_dir, "..", "larger_sets", "videos_even_larger2"))
    os.makedirs(data_folder_actuator, exist_ok=True)
    os.makedirs(video_folder_actuator, exist_ok=True)

    # --- Determine the starting trajectory index ---
    max_index_actuator = get_max_index(data_folder_actuator)
    start_index = max_index_actuator + 1
    print(f"Starting trajectory index: {start_index}")

    # --- Define CSV Column Names ---
    geom_columns = [f"{axis}_{geom_names_to_include[i]}" for i in range(len(geom_names_to_include)) for axis in ['x', 'y', 'z']]
    columns_actuator = ["time", "x_ee", "y_ee", "z_ee"] + geom_columns
    if include_velocity:
        geom_velocity_columns = [f"{axis}_velocity_{geom_names_to_include[i]}" for i in range(len(geom_names_to_include)) for axis in ['x', 'y', 'z']]
        columns_actuator += ["x_velocity_ee", "y_velocity_ee", "z_velocity_ee"] + geom_velocity_columns
    control_columns = [f"control_{i}" for i in range(model.nu // 2)]
    columns_actuator += control_columns

    # --- Main Simulation Loop: Only Actuator Dynamics ---
    for traj in range(n_trajectories):
        traj_index = start_index + traj  # Continue numbering from the existing files
        print(f"Simulating actuator dynamics trajectory {traj_index} ...")

        traj_actuator, frames_actuator = simulate_actuator(
            initial_conditions[traj], initial_controls[traj], model, data, duration,
            include_velocity, geom_ids_to_include, steady_state_z_values,
            ee_site_id, framerate, create_video, lambda_matrix,
            detect_cnvergece, convergence_threshold, window_size
        )
        # --- Steady-State Offset Correction (Actuator) ---
        steady_state_offset = np.mean(traj_actuator[-3:], axis=0)
        print(f"Steady-state reference (actuator) for trajectory {traj_index}: {steady_state_offset}")
        for i in range(len(traj_actuator)):
            traj_actuator[i][1:] = np.array(traj_actuator[i][1:]) - steady_state_offset[1:]
        df_actuator = pd.DataFrame(traj_actuator, columns=columns_actuator)
        output_csv_file_actuator = os.path.join(data_folder_actuator, f"state_traj_{traj_index}.csv")
        df_actuator.to_csv(output_csv_file_actuator, index=False)
        print(f"Actuator dynamics trajectory {traj_index} saved to {output_csv_file_actuator}.")
        if create_video:
            output_video_file_actuator = os.path.join(video_folder_actuator, f"video_state_traj_{traj_index}.mp4")
            media.write_video(output_video_file_actuator, frames_actuator, fps=framerate)
            print(f"Actuator dynamics video {traj_index} saved to {output_video_file_actuator}.")

    print("Simulations finished for actuator dynamics.")

if __name__ == "__main__":
    main()
