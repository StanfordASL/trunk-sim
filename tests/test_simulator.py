import unittest
import os
import numpy as np

from trunk_sim.simulator import TrunkSimulator, get_model_path
from trunk_sim.generate_trunk_model import generate_trunk_model


class TestTrunkSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = TrunkSimulator()

    def test_load_model(self):
        self.assertIsNotNone(self.simulator.model, "Model should be loaded")

    def test_step_simulation(self):
        initial_state = self.simulator.get_states()
        self.simulator.step(np.ones(self.simulator.n_controls))
        new_state = self.simulator.get_states()
        assert not np.allclose(
            initial_state, new_state
        ), "State should change after step"


if __name__ == "__main__":
    unittest.main()
