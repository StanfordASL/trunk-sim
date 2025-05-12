import torch
import numpy as np
from typing import List, Optional

class TrunkPolicy:
    """
    Simple wrapper around a (custom) policy function.
    """

    def __init__(self, policy, select_links: Optional[List[int]] = None):
        self.policy = policy
        self.select_links = select_links

    def __call__(self, t, state):
        filtered_state = state if self.select_links is None else state[self.select_links, :]
        control_inputs = self.policy(t, filtered_state)
        
        return control_inputs


class HarmonicPolicy(TrunkPolicy):
    """
    Simple periodic policy that returns a constant control input.
    """

    def __init__(self, frequency_range, amplitude_range, phase_range, num_segments=3):
        """
        Initialize the policy with a given discrete-time frequency and amplitude.
        """

        if len(frequency_range) != num_segments:
            frequency_range = [frequency_range for _ in range(num_segments)]
        
        if len(amplitude_range) != num_segments:
            amplitude_range = [amplitude_range for _ in range(num_segments)]

        if len(phase_range) != num_segments:
            phase_range = [phase_range for _ in range(num_segments)]

        self.frequencies = [np.random.uniform(*frequency_range[i]) for i in range(num_segments)]
        self.amplitudes = [np.random.uniform(*amplitude_range[i]) for i in range(num_segments)]
        self.phases = [np.random.uniform(*phase_range[i]) for i in range(num_segments)]
        self.signs = [np.random.choice([-1, 1]) for _ in range(num_segments)]
        self.policy = lambda t, _: np.array([[
            self.amplitudes[i] * np.sin(self.signs[i] * 2 * np.pi * self.frequencies[i] * t + self.phases[i]),
            self.amplitudes[i] * np.cos(self.signs[i] * 2 * np.pi * self.frequencies[i] * t + self.phases[i]),
            ] for i in range(num_segments)])

        super().__init__(self.policy)

class RandomWalkPolicy(TrunkPolicy):
    """
    Simple random policy that returns a random control input.
    """

    def __init__(self, num_segments=3, max_amplitude=12.0, dt=0.1):
        self.max_amplitude = max_amplitude
        self.dt = dt

        self.input = np.zeros((num_segments, 2))
        self.t = -np.inf
        super().__init__(self._policy)

    def _policy(self, t, _):
        if t <= self.t + self.dt:
            return self.input
        
        delta_input = np.sqrt(self.dt) * np.random.normal(size=(self.input.shape))
        new_input = self.input + delta_input
        new_input = np.clip(new_input, -self.max_amplitude, self.max_amplitude)
        self.input = new_input

        return new_input

def steady_state_input(num_segments, num_controls_per_segment=2, amplitude=1.0, angle=1):
    print(f"steady_state_input: num_segments={num_segments}, num_controls_per_segment={num_controls_per_segment}, amplitude={amplitude}, angle={angle}")
    """
    Get a steady state control input for a given number of segments and controls per segment.
    """
    assert num_controls_per_segment == 2, "Only implemented for 2 controls per segment"

    vec = np.array([np.cos(angle), np.sin(angle)])

    return np.vstack([vec for _ in range(num_segments)]) * amplitude
