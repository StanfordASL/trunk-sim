import mujoco
import os

from trunk_sim import utils

import os
os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mediapy as media
from typing import List, Callable, Any, Optional
import numpy as np

def load_mujoco_model_from_file(model_file: str) -> mujoco.MjModel:
    """
    Load a Mujoco model from a file.
    """
    os.environ["MUJOCO_GL"] = "egl"
    with open(model_file, "r") as f:
        model_xml = f.read()

    model = mujoco.MjModel.from_xml_string(model_xml)
    return model


def create_data_from_mujoco_model(model: mujoco.MjModel) -> mujoco.MjData:
    """
    Create a Mujoco data object from a Mujoco model.
    """
    data = mujoco.MjData(model)
    
    mujoco.mj_kinematics(model, data)

    return data

def set_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray):
    """
    Set the state of a Mujoco model.
    """
    mujoco.mj_resetData(model, data)  # Reset state and time.
    data.qpos[:] = qpos
    data.qvel[:] = qvel

def render_mujoco_model(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Render a Mujoco model.
    """
    with mujoco.Renderer(model) as renderer:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        media.show_image(renderer.render())


def simulate_mujoco_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    num_steps: int = 1000,
    timestep_ms: float = 10,
    render_video: bool = False,
    framerate_hz: float = 30,
) -> SimData:
    """
    Simulate a Mujoco model.
    """
    duration_s = num_steps * timestep_ms / 1000

    frames = []
    sim_data = SimData()
    
    iters = 0

    with mujoco.Renderer(model) as renderer:
        while data.time < duration_s:
            if np.abs(iters*timestep_ms - data.time*1e3) < 1e-2:
                sim_data.add_data(data)
                iters += 1

            if render_video and len(frames) < data.time * framerate_hz:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

            mujoco.mj_step(model, data)
    
    assert iters*timestep_ms >= data.time*1e3 - 1, "Simulation duration does not match stored data. Verify timestep_ms."

    if render_video:
        media.show_video(frames, fps=framerate_hz)

    return sim_data

def update_mujoco_data_from_torch_data(mujoco_data, torch_data):
    mujoco_data.qpos = np.hstack([np.hstack([torch_data.x[i,:3],np.zeros(4)]) for i in range(torch_data.x.size(0))])
    mujoco_data.qvel = np.hstack([np.hstack([torch_data.x[i,3:],np.zeros(3)]) for i in range(torch_data.x.size(0))])
    return mujoco_data


def play_mujoco_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    data_list,
    file_path,
    timestep_ms: float = 10,
    framerate_hz: float = 30,
) -> SimData:
    """
    Simulate a Mujoco model.
    """
    duration_s = len(data_list) * timestep_ms / 1000

    frames = []
    
    iters = 0
    data_list_iter = iter(data_list)

    mujoco.mj_resetData(model, data)

    with mujoco.Renderer(model) as renderer:
        while data.time < duration_s:
            if len(frames) < data.time * framerate_hz:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)

            if np.abs(iters*timestep_ms - data.time*1e3) < 1e-6:
                try:
                    data = update_mujoco_data_from_torch_data(data,next(data_list_iter))
                    iters += 1
                except StopIteration:
                    break

            mujoco.mj_forward(model, data)
            data.time += 0.02

    assert len(frames) > 0, "No frames were generated"
    
    return media.write_video(file_path, frames)



class TrunkSimulator:
    def __init__(self, model_path, dt=0.01): # dt correponds to state and input update rate, not the simulation timestep
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.timestep = timestep
        self.sim_dt = 
        self.policy = None

        # Store states and inputs 
        self.states = []
        self.inputs = []

        self.reset()
    
    

    def reset(self, qpos=None, qvel=None):
        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.

    def set_state(self, qpos, qvel):
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel

    def step():
        for i in range(10):
        mujoco.mj_step(self.model, self.data)

    def has_converged():

    def set_initial_state():
        pass


    def simulate(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        num_steps: int = 1000,
        timestep_ms: float = 10,
        render_video: bool = False,
        framerate_hz: float = 30,
        polciy: Callable[[Any], Any] = None,
    ) -> SimData:
        """
        Simulate a Mujoco model.
        """
        duration_s = num_steps * timestep_ms / 1000

        frames = []
        sim_data = SimData()
        
        iters = 0

        with mujoco.Renderer(model) as renderer:
            while data.time < duration_s:
                if np.abs(iters*timestep_ms - data.time*1e3) < 1e-2:
                    sim_data.add_data(data)
                    iters += 1

                if render_video and len(frames) < data.time * framerate_hz:
                    renderer.update_scene(data)
                    pixels = renderer.render()
                    frames.append(pixels)

                mujoco.mj_step(model, data)
        
        assert iters*timestep_ms >= data.time*1e3 - 1, "Simulation duration does not match stored data. Verify timestep_ms."

        if render_video:
            media.show_video(frames, fps=framerate_hz)

        return sim_data
    
    def get_states():
        pass


# Example usage
if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'trunk_model.xml')
    simulator = MuJoCoSimulator(model_path)
    
    simulator.reset()
    for _ in range(1000):
        simulator.step()
        simulator.render()
    simulator.close()