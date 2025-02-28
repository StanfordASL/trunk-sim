import mujoco
import numpy as np
import mediapy as media
from typing import Tuple, Optional

from trunk_sim.simulator import TrunkSimulator
from trunk_sim.policy import TrunkPolicy
from trunk_sim.data import TrunkData


def rollout(simulator: TrunkSimulator,
            policy: Optional[TrunkPolicy] = None,
            data: Optional[TrunkData] = None,
            initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            duration: float = 1.0,  # [s]
            render_video: bool = False,
            framerate: int = 30,  # [Hz]
            video_filename: Optional[str] = "trunk_render.mp4",
            stop_at_convergence: bool = False) -> None:
    """
    Rollout a policy on a simulator and save it inside a data object.
    """

    simulator.reset_time()
    
    if render_video:
        frames = []

        with mujoco.Renderer(simulator.model) as renderer:
            while simulator.data.time < duration and (not stop_at_convergence or not simulator.has_converged()):
                rollout_step(simulator, policy, data)

                # Rendering
                renderer.update_scene(simulator.data)
                pixels = renderer.render()
                frames.append(pixels)

        media.write_video(video_filename, frames, fps=framerate)
        
    else:
        while simulator.data.time < duration and (not stop_at_convergence or not simulator.has_converged()):
            rollout_step(simulator, policy, data)


def rollout_step(simulator: TrunkSimulator,
                 policy: Optional[TrunkPolicy] = None,
                 data: Optional[TrunkData] = None) -> None:
    """
    Perform a single step of a rollout.
    """
    state = simulator.get_states()
    t = simulator.get_time()
    control_input = policy(t, state) if policy is not None else None
        
    t, x, u, x_new = simulator.step(control_input)

    if data is not None:
        data.add_data(t, x, u, x_new)