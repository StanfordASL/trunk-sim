import mujoco
import numpy as np
import mediapy as media
from typing import Tuple, Optional

from trunk_sim.simulator import TrunkSimulator
from trunk_sim.policy import TrunkPolicy
from trunk_sim.data import TrunkData


def rollout(data: TrunkData,
            simulator: TrunkSimulator,
            policy: TrunkPolicy = None,
            num_rollouts: int = 1,
            initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            duration: float = 1.0,  # [s]
            render_video: bool = False,
            framerate: int = 30,  # [Hz]
            video_filename: Optional[str] = "trunk_render.mpy"):
    """
    Rollout a policy on a simulator and save it inside a data object.
    """
    # TODO: perhaps you just want a single rollout here because you probably want diverse initial states
    # Also, currently we would save many videos if render_video is True to the same file
    for _ in range(num_rollouts):
        simulator.reset()
        if initial_state is not None:
            simulator.set_state(*initial_state)
        state = simulator.get_state()
        converged = False
        
        if render_video:
            frames = []

            with mujoco.Renderer(simulator.model) as renderer:
                while simulator.data.time < duration and not converged:
                    control_input = policy(state)
                    t, state_new, converged = simulator.step(control_input)
                    data.add_data(t, state, control_input)
                    
                    # Rendering
                    renderer.update_scene(simulator.data)
                    pixels = renderer.render()
                    frames.append(pixels)

                    # Update state
                    state = state_new

            media.write_video(video_filename, frames, fps=framerate)
            
        else:
            while simulator.data.time < duration and not converged:
                control_input = policy(state)
                t, state_new, converged = simulator.step(control_input)
                data.add_data(t, state, control_input)

                # Update state
                state = state_new
