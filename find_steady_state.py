import mujoco
import mediapy as media
import numpy as np
import pandas as pd
import os

# Parameters
duration = 15.0  # Run simulation for this duration to ensure steady state (seconds)
framerate = 60  # Frames per second for video rendering
current_dir = os.path.dirname(__file__)
video_output_path = os.path.join(current_dir, "trajectories", "videos", "steady_state_video.mp4")
csv_output_path = os.path.join(current_dir, "trajectories", "data", "steady_state_z_positions.csv")

# Model and data setup
xml_path = os.path.join(current_dir, "models", "cable_trunk_expanded_old.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Exclude the world body (' ') from the geom list
geom_names = [f"actuatedG{i}" for i in range(model.ngeom)]  # 'actuatedGi' are the geom names
geom_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    for geom_name in geom_names
]

# End-effector site ID
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

# Run the simulation
frames = []

print("Starting simulation...")
with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)

        # Render video frame
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

# print slider joint steady state q
print("slider joint steady state:", data.qpos[0])            

# Extract final z-coordinates (steady state)
print("Extracting steady-state z-coordinates...")
ee_z = [data.site_xpos[ee_site_id][2]]  # End-effector z-coordinate
geom_z_positions = [data.geom_xpos[geom_id][2] for geom_id in geom_ids]  # Geom z-coordinates

# Prepare data for saving
steady_state_data = ee_z + geom_z_positions

# Save video
os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
media.write_video(video_output_path, frames, fps=framerate)
print(f"Video saved to: {video_output_path}")

# Save z-coordinates to CSV
os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
columns = ["ee_z"] + [f"{geom_names[i]}_z" for i in geom_ids]
df = pd.DataFrame([steady_state_data], columns=columns)
df.to_csv(csv_output_path, index=False)
print(f"Steady-state z-coordinates saved to: {csv_output_path}")

