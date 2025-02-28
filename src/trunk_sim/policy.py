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
                    raise ValueError(f"Policy evaluation failed with both numpy and torch formats: {e}, {e2}")
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
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

class HarmonicPolicy(TrunkPolicy):
    """
    Simple periodic policy that returns a constant control input.
    """
    def __init__(self, frequency=1.0, amplitude=1.0, phase=0.0, num_segments=3):
        """
        Initialize the policy with a given discrete-time frequency and amplitude.
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase
        self.policy = lambda t, _: amplitude * np.array([np.sin(2 * np.pi * frequency * t + self.phase), np.cos(2 * np.pi * frequency * t + self.phase)] * num_segments)

        super().__init__(self.policy)