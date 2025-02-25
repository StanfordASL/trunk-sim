import mujoco
import glfw
import numpy as np

# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

# Create a window and make OpenGL context current
window = glfw.create_window(1200, 900, "MuJoCo Visualization", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)  # Enable v-sync

# Load the model and create data
xml_path = "/Users/paulleonardwolff/Desktop/Stanford/Simulation_Python/Relevant_Files_Mujoco_trunk/models/cable_trunk_expanded_old_4_tendons.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Visualization data structures
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, maxgeom=1000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

# Initialize visualization settings
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

# Highlight specific geoms and sites
highlight_geoms = ["actuatedG0", "actuatedG15", "actuatedG25", "actuatedG32", "actuatedG37"]
highlight_sites = ["endeffector"]

# Main loop
while not glfw.window_should_close(window):
    # Advance simulation for 1/60 second
    simstart = data.time
    while data.time - simstart < 1.0 / 60.0:
        mujoco.mj_step(model, data)

    # Get the framebuffer viewport
    width, height = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, width, height)

    # Update the scene
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

    # Highlight selected geoms
    for geom_name in highlight_geoms:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id != -1:
            scn.geoms[geom_id].rgba = np.array([1, 0, 0, 1], dtype=np.float32)  # Red for selected geoms

    # Highlight selected sites by adding temporary geoms
    for site_name in highlight_sites:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id != -1:
            site_pos = np.array(data.site_xpos[site_id], dtype=np.float64)  # Ensure it's a proper NumPy array
            temp_geom = mujoco.MjvGeom()

            # Correct transformation matrix: must be 9 elements, row-major order
            identity_mat = np.eye(3, dtype=np.float64).flatten()

            mujoco.mjv_initGeom(
                temp_geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.01, 0.01, 0.01], dtype=np.float64),  # Sphere size
                site_pos,
                identity_mat,  # 9-element row-major matrix
                np.array([0, 1, 0, 1], dtype=np.float32)  # Green RGBA color
            )

            # Add the temporary geom to the scene
            scn.geoms.append(temp_geom)

    # Render the scene
    mujoco.mjr_render(viewport, scn, con)

    # Swap buffers and poll events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Cleanup
glfw.terminate()
scn.free()
con.free()
