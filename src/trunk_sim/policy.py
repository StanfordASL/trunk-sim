import torch
import numpy as np


class TrunkPolicy:
    """
    Simple wrapper around a (custom) policy function.
    """

    def __init__(self, policy):
        self.policy = policy
        self._is_torch_policy = None

    def __call__(self, t, state):
        """
        Get the control inputs for a given state.
        First tries with the original state type, then converts to torch tensor if needed.

        Args:
            state: The current state (numpy array or torch tensor)

        Returns:
            control_inputs: Control inputs as numpy array
        """
        if self._is_torch_policy is None:
            # If we don't know the policy type yet, try to determine it
            try:
                control_inputs = self.policy(t, state)
                self._is_torch_policy = isinstance(control_inputs, torch.Tensor)
            except (TypeError, ValueError, RuntimeError) as e:
                # Failed with original state format, try with torch tensor
                try:
                    if not isinstance(state, torch.Tensor):
                        state = torch.tensor(state, dtype=torch.float32)
                    control_inputs = self.policy(t, state)
                    self._is_torch_policy = True
                except Exception as e2:
                    raise ValueError(
                        f"Policy evaluation failed with both numpy and torch formats: {e}, {e2}"
                    )
        else:
            # We already know the policy type, use the appropriate format
            if self._is_torch_policy and not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32)

            control_inputs = self.policy(t, state)

        # Convert the result to numpy if it's a torch tensor
        if isinstance(control_inputs, torch.Tensor):
            control_inputs = control_inputs.detach().cpu().numpy()

        return control_inputs

    def reset(self):
        """
        Reset the policy if it has an internal state.
        Attempts to call a reset method if available.
        """
        if hasattr(self.policy, "reset"):
            self.policy.reset()


class HarmonicPolicy(TrunkPolicy):
    """
    Simple periodic policy that returns a constant control input.
    """

    def __init__(self, frequency_range, amplitude_range, phase_range, num_segments=3):
        """
        Initialize the policy with a given discrete-time frequency and amplitude.
        """
        self.frequencies = [np.random.uniform(*frequency_range) for _ in range(num_segments)]
        self.amplitudes = [np.random.uniform(*amplitude_range) for _ in range(num_segments)]
        self.phases = [np.random.uniform(*phase_range) for _ in range(num_segments)]
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

def steady_state_input(num_segments, num_controls_per_segment=2, amplitude=1.0, angle=1, verbose=False):
    """
    Get a steady state control input for a given number of segments and controls per segment.
    """
    assert num_controls_per_segment == 2, "Only implemented for 2 controls per segment"

    vec = np.array([np.cos(angle), np.sin(angle)])
    
    if verbose:
        print(f"steady_state_input: num_segments={num_segments}, num_controls_per_segment={num_controls_per_segment}, amplitude={amplitude}, angle={angle}")

    return np.vstack([vec for _ in range(num_segments)]) * amplitude
