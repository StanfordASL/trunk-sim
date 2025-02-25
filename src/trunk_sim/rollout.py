

def rollout(simulator: TrunkSimulator, policy: Optional[Policy] = None, num_rollouts: int = 1, initial_state: Optional[Tuple[np.ndarray, np.ndarray]] = None, duration_s: float = 1.0, timestep_ms: float = 10, render_video: bool = False, framerate_hz: float = 30) -> List[SimData]:
    """
    Rollout a policy on a simulator.
    """
    simulator.set_initial_state()

    data = None
    
    with mujoco.Renderer(model) as renderer:
        for i in range(1000):
            single_pass(simulator, policy)


def single_pass(simulator: TrunkSimulator, policy, ):
    simulator.step(renderer)
    states = simulator.get_states()
    u = policy(states)
    simulator.set_control_input(u)
    data.add_