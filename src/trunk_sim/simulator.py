import mujoco
from typing import Optional
import numpy as np
import mediapy as media

def get_model_path(model_type: Optional[str] = "default") -> str:
    if model_type == "default":
        return "src/trunk_sim/models/cable_trunk_expanded_old_4_tendons.xml"
    else:
        raise ValueError("Model type not recognized.")

def render_simulator(simulator):
    """
    Render a Mujoco model.
    """
    with mujoco.Renderer(simulator.model) as renderer:
        mujoco.mj_forward(simulator.model, simulator.data)
        renderer.update_scene(simulator.data)
        media.show_image(renderer.render())

class TrunkSimulator:
    def __init__(self, model_path: str, timestep: Optional[float] = 0.01):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.timestep = timestep  # Measured state and input timestep
        self.sim_dt = 0.002  # TODO: Obtain from mujoco model. Corresponds to simulation timestep of mujoco model.

        self.sim_steps = self.timestep / self.sim_dt
        if self.sim_steps % 1 != 0:
            raise ValueError("Timestep must be a multiple of the simulation timestep.")
        else:
            self.sim_steps = int(self.sim_steps)

        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.
        mujoco.mj_kinematics(self.model, self.data) #TODO: Verify if this is necessary
        

    def set_state(self, qpos = None, qvel = None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel

    def step(self, control_input=None):
        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        return self.data.time, self.get_states()
    
    def has_converged():
        pass

    def get_states(self):
        return np.array(
            [self.data.body(b).xpos.copy().tolist() for b in range(1,self.model.nbody)]
        )

    def set_control_input(self, control_input):
        pass
