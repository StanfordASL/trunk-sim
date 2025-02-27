import mujoco
from typing import Optional
import numpy as np
import mediapy as media

from trunk_sim.generate_trunk_model import generate_trunk_model


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


class Simulator:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_xml: Optional[str] = None,
        timestep: Optional[float] = 0.01,
    ):
        # Load model
        if model_xml and not model_path:
            self.model = mujoco.MjModel.from_xml_string(model_xml)
        elif model_path and not model_xml:
            self.model_path = model_path
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
        else:
            raise ValueError("Either model_path or model_xml must be provided.")

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
        mujoco.mj_kinematics(self.model, self.data)  # TODO: Verify if this is necessary

    def set_state(self, qpos=None, qvel=None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel

    def step(self, control_input=None):
        self.set_control_input(control_input)
        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        return self.data.time, self.get_states()

    def has_converged(self, threshold=1e-6):
        if self.prev_states is None:
            self.prev_states = self.get_states()
            return False
        if np.linalg.norm(self.prev_states - self.get_states()) < threshold:
            return True
        else:
            return False

    def get_states(self):
        return np.array(
            [self.data.body(b).xpos.copy().tolist() for b in range(1, self.model.nbody)]
        )

    def set_control_input(self, control_input=None):
        if control_input is not None:
            self.data.ctrl[:] = control_input
        else:
            self.data.ctrl[:] = 0


class TrunkSimulator(Simulator):
    def __init__(
        self,
        n_links: int = 100,
        payload_mass: float = 0.1,
        timestep: Optional[float] = 0.01,
    ):
        super().__init__(
            model_xml=generate_trunk_model(n_links=n_links, payload_mass=payload_mass),
            timestep=timestep,
        )

        self.B = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) # Mapping from control input to actuators

    def set_control_input(self, control_input=None):
        if control_input is not None:
            self.data.ctrl[:] = self.B @ control_input
        else:
            self.data.ctrl[:] = 0
