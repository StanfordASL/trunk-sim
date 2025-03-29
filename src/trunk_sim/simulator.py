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
        track_bodies: Optional[list] = None,
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
        self.sim_timestep = 0.002  # Mujoco simulation timestep

        assert (
            self.sim_timestep <= self.timestep
        ), "Timestep must be greater than Mujoco timestep."

        self.sim_steps = self.timestep / self.sim_timestep
        if self.sim_steps % 1 != 0:
            raise ValueError("Timestep must be a multiple of the simulation timestep.")
        else:
            self.sim_steps = int(self.sim_steps)

        if track_bodies:
            self.track_bodies = track_bodies
        else:
            self.track_bodies = range(self.model.nbody)

        self.reset()
    
    def reset(self):
        self.prev_states = None
        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.
        #mujoco.mj_kinematics(self.model, self.data)  # TODO: Verify if this is necessary

        self.positions = None
        self.prev_positions = None
        self._set_states()

    def reset_time(self):
        self.data.time = 0

    def set_state(self, qpos=None, qvel=None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel

    def has_converged(self, threshold=1e-3):
        if self.prev_states is None:
            self.prev_states = self.get_states()
            return False

        if np.linalg.norm(self.prev_states - self.get_states()) < threshold:
            return True
        else:
            return False

    def get_states(self):
        # TODO: Mujoco supports getting velocities directly but requires coordinate transformation
        return np.concatenate([self.positions, self.velocities], axis=1)
    
    def get_current_positions(self):
        return np.array(
            [self.data.body(b).xpos.copy().tolist() for b in self.track_bodies]
        )
    
    def _set_states(self):
        self.positions = self.get_current_positions()

        # Reporting initial velocities as zero
        # Valid since the system is at rest at the beginning
        if self.prev_positions is None:
            self.prev_positions = self.positions

        # Calculate velocity as v_k = (x_{k} - x_{k-1}) / dt but with short time steps from mujoco
        self.velocities = (self.positions - self.prev_positions) / self.sim_timestep
    
    def get_time(self):
        return self.data.time

    def set_control_input(self, control_input=None):
        if control_input is not None:
            self.data.ctrl[:] = control_input
        else:
            self.data.ctrl[:] = 0

        return self.data.ctrl

    def set_initial_steady_state(self, steady_state_control_input, kick=None, max_duration=10.0, kick_duration=0.01):
        self.reset()

        while not self.has_converged() and self.data.time < max_duration:
            self.step(steady_state_control_input)
            #print(self.data.actuator_force)

        current_time = self.data.time
        if kick is not None:
            while self.data.time < current_time + kick_duration:
                self.step(kick)

    def step(self, control_input=None):
        t = self.get_time()
        x = self.get_states()
        u = self.set_control_input(control_input)

        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

            if i == self.sim_steps - 2:
                self.prev_positions = self.get_current_positions()

        self._set_states()

        x_new = self.get_states()

        return t, x, u, x_new
            
class TrunkSimulator(Simulator):
    def __init__(
        self,
        num_segments: int = 3,
        num_links_per_segment: int = 10,
        tip_mass: float = 0.5,
        timestep: Optional[float] = 0.01,
        radius: float = 0.025,
        length: float = 0.32,
        spacing: float = 2.0,
        tip_size: Optional[float] = None,
    ):

        self.num_controls_per_segment = 2
        self.num_controls_per_segment_mujoco = 4
        self.num_segments = num_segments
        self.num_links_per_segment = num_links_per_segment
        self.num_links = num_segments * num_links_per_segment

        # Mapping from control input to mujoco actuators
        self.input_map = np.array(
            [[1, 0], [-1, 0], [0, 1], [0, -1]]
        )  # (num_controls_per_segment_mujoco x num_controls_per_segment)

        self.inverse_input_map = np.linalg.pinv(
            self.input_map
        )  # (num_controls_per_segment x num_controls_per_segment_mujoco)

        super().__init__(
            model_xml=generate_trunk_model(
                num_segments=num_segments,
                num_links_per_segment=num_links_per_segment,
                tip_mass=tip_mass,
                radius=radius,
                length=length,
                spacing=spacing,
                tip_size=tip_size,
            ),
            timestep=timestep,
        )

        self.track_bodies = self.get_link_bodies()
        self.reset()

    def get_link_bodies(self):
        track_bodies = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if "link" in name:
                track_bodies.append(i)
                
        return track_bodies
    
    def set_control_input(self, control_input=None):
        """
        control_input: np.array of shape (num_segments, num_controls_per_segment)
        """

        # shape of u_mujoco: (num_segments x num_controls_per_segment_mujoco).flatten()
        u_mujoco = (
            (control_input @ self.input_map.T).flatten()
            if control_input is not None
            else None
        )
        u = (
            super()
            .set_control_input(u_mujoco)
            .reshape(-1, self.num_controls_per_segment_mujoco)
            @ self.inverse_input_map.T
        )

        return u

