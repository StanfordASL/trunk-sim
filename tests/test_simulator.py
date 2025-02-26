import unittest
import os
import numpy as np

from trunk_sim.simulator import TrunkSimulator, get_model_path

class TestTrunkSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = TrunkSimulator(get_model_path(), timestep=1.0)
        self.simulator.set_state(qvel=np.random.randn(len(self.simulator.data.qvel))) # Random initial state

    def test_load_model(self):
        self.assertIsNotNone(self.simulator.model, "Model should be loaded")

    def test_step_simulation(self):
        initial_state = self.simulator.get_states()
        self.simulator.step()
        new_state = self.simulator.get_states()
        assert not np.allclose(initial_state, new_state), "State should change after step"

if __name__ == '__main__':
    unittest.main()