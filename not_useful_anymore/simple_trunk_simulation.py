import mujoco
import mediapy as media
import numpy as np
import os

print(mujoco.__version__)

current_dir = os.path.dirname(__file__)
xml_path = os.path.join(current_dir, "models", "cable_trunk_expanded_old_4_tendons.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "endeffector")

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# define control callback that randomly actuates tendons
def random_controller(mjModel, mjData):
  mjData.ctrl[:] = 50*np.random.uniform(-1.0, 1.0, mjModel.nu)

mujoco.set_mjcb_control(lambda m, d: random_controller(m, d)) 

joint_type_map = {0: "free", 1: "ball", 2: "slider", 3: "hinge"}

for dof_index in range(model.nv):
    joint_id = model.dof_jntid[dof_index]
    joint_name = model.names[model.name_jntadr[joint_id]]
    joint_type = joint_type_map[model.jnt_type[joint_id]]
    print(f"DOF {dof_index} corresponds to joint: {joint_name} (Type: {joint_type})")

names = [model.geom(i).name for i in range(model.ngeom)]

# Simulate and display video.
frames = []
with mujoco.Renderer(model) as renderer:
  while data.time < duration:
    mujoco.mj_step(model, data)
    #print(data.site_xpos[ee_site_id])
    print(data.geom_xpos)
    max_qpos_idx = np.argmax(data.qpos)# find maximum index
    #print(max_qpos_idx)
    # print(data.ctrl)
    # print(model.nu)
    # print(model.nv)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data)
      pixels = renderer.render()
      frames.append(pixels)

print('Simulation done')
video_folder = os.path.join(current_dir, "trajectories", "videos", "output_video.mp4")
media.write_video(video_folder, frames, fps=framerate)
print('Rendering done')