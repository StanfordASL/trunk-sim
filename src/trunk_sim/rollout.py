from typing import List, Optional, Tuple
import numpy as np
import mujoco
import mediapy as media

from trunk_sim.simulator import TrunkSimulator

framerate_hz = 30

def rollout(simulator: TrunkSimulator, policy = None, num_rollouts: int = 1, initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None, duration_s: float = 1.0, timestep_ms: float = 10, render_video: bool = False, video_filename: Optional[str] = "render.mpy"):
    """
    Rollout a policy on a simulator.
    """

    #TODO: Apply num_rollouts, but how do we handle data and initial_state?

    simulator.reset()
    simulator.set_state(*initial_state)
    data = None #TODO: Hugo
    
    if render_video:
        frames = []
        framerate_hz = int(1.0/timestep_ms) # TODO: Make independent of timestep_ms

        with mujoco.Renderer(simulator.model) as renderer:
            while simulator.data.time < duration_s:
                single_pass(simulator, policy, data)
                renderer.update_scene(simulator.data)
                pixels = renderer.render()
                frames.append(pixels)

        media.write_video(video_filename, frames, fps=framerate_hz)
        
    else:
        while simulator.data.time < duration_s:
            single_pass(simulator, policy, data)


def single_pass(simulator: TrunkSimulator, policy, data):
    states = simulator.get_states()
    #u = set_control_input(policy(states)) #TODO: Hugo
    simulator.step(u)
    new_state = simulator.get_states()
    #data.append(simulator.get_states())