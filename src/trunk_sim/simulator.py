import mujoco
from typing import Optional
import numpy as np


def get_model_path(model_type: str) -> str:
    if model_type == "default":
        return "src/models/cable_trunk_expanded_old_4_tendons.xml"
    else:
        raise ValueError("Model type not recognized.")


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
        # mujoco.mj_kinematics(model, data) TODO: Verify if this is necessary
        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.

    def set_state(self, qpos, qvel):
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel

    def step(self):
        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

    def has_converged():
        pass

    def get_states(self):
        return self.data.qpos, self.data.qvel

    def set_control_input(self, control_input):
        pass
