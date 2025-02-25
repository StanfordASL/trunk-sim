import mujoco 
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def apply_random_control(mjData, mjmodel, duration, scale, update_frequency):
    timestep = mjmodel.opt.timestep
    steps_per_update = int(1 / (update_frequency * timestep))
    end_time = mjData.time + duration

    last_applied_control = None

    while mjData.time < end_time:
        if int(mjData.time / timestep) % steps_per_update == 0:
            last_applied_control = np.random.uniform(-scale, scale, mjmodel.nu)  # Update control at specified frequency
            mjData.ctrl[:] = last_applied_control
        mujoco.mj_step(mjmodel, mjData)
    mjData.ctrl[:] = 0
    return last_applied_control

def apply_random_control_symmetric(mjData, mjmodel, duration, scale, update_frequency):
    timestep = mjmodel.opt.timestep
    steps_per_update = int(1 / (update_frequency * timestep))
    end_time = mjData.time + duration

    # Calculate the number of independent control inputs.
    assert mjmodel.nu % 2 == 0, "mjmodel.nu must be even."
    num_indep = mjmodel.nu // 2
    last_applied_control = None

    while mjData.time < end_time:
        if int(mjData.time / timestep) % steps_per_update == 0:
            # Generate a random control for the independent inputs.
            random_control = np.random.uniform(-scale, scale, num_indep)
            # Extend the control using the symmetry property:
            # Each independent input u_i is mapped to [u_i, -u_i]
            extended_control = np.concatenate([[u, -u] for u in random_control])
            last_applied_control = random_control
            mjData.ctrl[:] = extended_control
        mujoco.mj_step(mjmodel, mjData)
    mjData.ctrl[:] = 0
    return last_applied_control


def apply_random_control_symmetric_update(mjData, mjmodel, duration, update_apply_duration, scale, update_limit,
                                          initial_mean, initial_noise_range, lambda_matrix):
    start_time = mjData.time
    hold_time = start_time + duration - update_apply_duration
    end_time = start_time + duration

    # Ensure even number of control inputs.
    assert mjmodel.nu % 2 == 0, "mjmodel.nu must be even."
    num_indep = mjmodel.nu // 2

    # For this update, we assume num_indep == 2.
    # Sample a radius and an angle.
    r = initial_mean + np.random.uniform(-initial_noise_range, initial_noise_range)
    angle = np.random.uniform(0, 2 * np.pi)
    current_control_setpoint = np.array([r * np.cos(angle), r * np.sin(angle)])
    current_control_setpoint = np.clip(current_control_setpoint, -scale, scale)
    last_applied_control_setpoint = current_control_setpoint.copy()

    # Initialize the actual control (u_real) as zero.
    current_control_real = np.zeros(num_indep)
    last_applied_control_real = current_control_real.copy()

    # Immediately apply the symmetric control based on u_real.
    extended_control = np.concatenate([[u, -u] for u in current_control_real])
    mjData.ctrl[:] = extended_control

    # Flag to ensure we update the setpoint only once.
    updated_u_input = False

    # We'll integrate the dynamics of u_real between simulation steps.
    t_prev = mjData.time

    # Define the ODE: du_real/dt = lambda_matrix * (u_real - u_setpoint)
    def ode_func(t, u, u_setpoint, lambda_matrix):
        return lambda_matrix.dot(u - u_setpoint)

    while mjData.time < end_time:
        t_new = mjData.time
        # Integrate from t_prev to t_new (with u_setpoint assumed constant over the interval)
        sol = solve_ivp(ode_func, [t_prev, t_new], current_control_real,
                        args=(current_control_setpoint, lambda_matrix), method="RK45")
        current_control_real = sol.y[:, -1]
        t_prev = t_new

        if mjData.time >= hold_time and not updated_u_input:
            # Instead of adding a delta vector, update the setpoint by changing its angle.
            # Compute the current angle and radius.
            theta = np.arctan2(current_control_setpoint[1], current_control_setpoint[0])
            r_current = np.linalg.norm(current_control_setpoint)
            # Sample a delta angle (update_limit now is an angular limit in radians).
            base_delta = np.random.choice([-3 * np.pi / 8, 3 * np.pi / 8])
            additional_delta = np.random.uniform(-np.pi / 8, np.pi / 8)
            delta_angle = base_delta + additional_delta
            new_theta = theta + delta_angle
            new_control_setpoint = np.array([r_current * np.cos(new_theta), r_current * np.sin(new_theta)])
            new_control_setpoint = np.clip(new_control_setpoint, -scale, scale)
            current_control_setpoint = new_control_setpoint
            updated_u_input = True

        last_applied_control_setpoint = current_control_setpoint.copy()
        last_applied_control_real = current_control_real.copy()

        # Apply the symmetric control using the real control u_real.
        extended_control = np.concatenate([[u, -u] for u in current_control_real])
        mjData.ctrl[:] = extended_control

        mujoco.mj_step(mjmodel, mjData)

    # mjData.ctrl[:] = 0
    return last_applied_control_setpoint, last_applied_control_real


def generate_invertible_matrix(n):
    """Generate an invertible matrix of size n x n."""
    while True:
        V = np.random.rand(n, n)
        if np.linalg.det(V) != 0:  # Check determinant
            return V

def generate_matrix_with_complex_eigenvalues(eigenvalues):
    """
    Generate a real matrix with the specified (complex conjugate pair) eigenvalues.
    Args:
        eigenvalues (list of complex): Desired eigenvalues (must include conjugate pairs for real matrix).
    Returns:
        np.ndarray: A real matrix with the specified eigenvalues.
    """
    n = len(eigenvalues)

    if not all(np.iscomplex(ev) or np.isreal(ev) for ev in eigenvalues):
        raise ValueError("Eigenvalues must be real or come in conjugate pairs.")

    # Step 1: Create the block-diagonal matrix with eigenvalues
    lambda_matrix = np.zeros((n, n), dtype=complex)
    i = 0
    while i < n:
        if np.iscomplex(eigenvalues[i]):
            # Create 2x2 block for conjugate pair
            a = eigenvalues[i].real
            b = eigenvalues[i].imag
            lambda_matrix[i:i + 2, i:i + 2] = np.array([[a, -b],
                                                        [b, a]])
            i += 2  # Skip the conjugate
        else:
            lambda_matrix[i, i] = eigenvalues[i]
            i += 1

    # Step 2: Generate a random invertible matrix
    V = np.eye(n)  # generate_invertible_matrix(n)
    V_inv = np.linalg.inv(V)

    # Step 3: Construct the target matrix
    A = np.real(V @ lambda_matrix @ V_inv)  # Ensure real part for the result
    return A

def compute_u_step(step_time, time, desired_u, lambda_matrix, num_controls):
    if time < step_time:
        current_u = np.zeros(num_controls)
    else:
        time_since_step = time - step_time
        current_u = desired_u + expm(lambda_matrix * time_since_step) @ (np.zeros(num_controls) - desired_u)

    return current_u


def compute_u_circle(time, amplitude, frequency, lambda_matrix, num_controls, last_time, u_last, step_time):

    if time < step_time:
        return np.zeros(num_controls), np.zeros(num_controls)

    # Check if the integration interval is zero-length.
    if time == last_time:
        # Compute the current setpoint using the current time.
        phase_shifts_current = np.linspace(0, np.pi / 2 * (num_controls - 1), num_controls)
        u_setpoint_current = amplitude * np.sin(2 * np.pi * frequency * time + phase_shifts_current)
        # Swap the second and third entries if applicable.
        if num_controls > 2:
            u_setpoint_current[1], u_setpoint_current[2] = u_setpoint_current[2], u_setpoint_current[1]
        return u_last, u_setpoint_current

    # Define the real dynamics function.
    def u_dynamics(t, u):
        phase_shifts = np.linspace(0, np.pi / 2 * (num_controls - 1), num_controls)
        # Note: Ensure that you use the same frequency convention as elsewhere.
        u_setpoint = amplitude * np.sin(2 * np.pi * frequency * t + phase_shifts)
        # Swap the second and third entries in the setpoint.
        if num_controls > 2:
            u_setpoint[1], u_setpoint[2] = u_setpoint[2], u_setpoint[1]
        return lambda_matrix @ (u - u_setpoint)

    # Numerically integrate from the previous time step to the current time.
    solution = solve_ivp(
        fun=u_dynamics,
        t_span=(last_time, time),
        y0=u_last,
        t_eval=[time],
        method="RK45"
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    phase_shifts_current = np.linspace(0, np.pi / 2 * (num_controls - 1), num_controls)
    u_setpoint_current = amplitude * np.sin(2 * np.pi * frequency * time + phase_shifts_current)
    if num_controls > 2:
        u_setpoint_current[1], u_setpoint_current[2] = u_setpoint_current[2], u_setpoint_current[1]

    sol_y = np.array(solution.y)
    return sol_y[:, -1], u_setpoint_current


from scipy.integrate import solve_ivp
import numpy as np


def compute_u_circle_sym(time, amplitude, frequency, lambda_matrix, num_controls, last_time, u_last, step_time):

    # Assert that we are in the 2-input (independent) case.
    # Assert that we are in the 2-input (independent) case.
    assert num_controls == 2, "This function is defined only for two independent controls."

    # If we're still in the initial phase, return zeros.
    if time < step_time:
        zeros = np.zeros(2)
        return zeros, zeros

    # If the integration interval is zero-length, simply compute the setpoint.
    if time == last_time:
        theta = 2 * np.pi * frequency * time
        u_setpoint_current = np.array([
            amplitude * np.sin(theta),
            amplitude * np.sin(theta + np.pi / 2)
        ])
        return u_last, u_setpoint_current

    # Define the dynamics for the independent control signals.
    def u_dynamics(t, u):
        theta = 2 * np.pi * frequency * t
        u_setpoint = np.array([
            amplitude * np.sin(theta),
            amplitude * np.sin(theta + np.pi / 2)
        ])
        # The dynamics are defined as du/dt = lambda_matrix @ (u - u_setpoint)
        return lambda_matrix @ (u - u_setpoint)

    # Numerically integrate from the previous time to the current time.
    solution = solve_ivp(
        fun=u_dynamics,
        t_span=(last_time, time),
        y0=u_last,
        t_eval=[time],
        method="RK45"
    )

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    theta = 2 * np.pi * frequency * time
    u_setpoint_current = np.array([
        amplitude * np.sin(theta),
        amplitude * np.sin(theta + np.pi / 2)
    ])

    u_updated = solution.y[:, -1]
    return u_updated, u_setpoint_current


def compute_u_dynamics_sin(
    u_last, t_last, t_current, _lambda, offsets, amplitudes, frequencies
):
    # Define the dynamics function
    def u_dynamics(t, u):
        # Calculate u_desired(t)
        u_desired = offsets + amplitudes * np.sin(frequencies * t)
        # Compute the derivative of u
        return _lambda @ (u - u_desired)

    if t_last == t_current:
        #print("t_last and t_current are the same.")
        return u_last

    # Numerically integrate the dynamics from t_last to t_current
    solution = solve_ivp(
        fun=u_dynamics,                      # Dynamics function
        t_span=(t_last, t_current),          # Time interval
        y0=u_last,                           # Initial value of u
        t_eval=[t_current],                  # Only evaluate at t_current
        method="RK45"                        # Numerical integration method
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    # Return the value of u at t_current
    #print("Type of solution.y:", type(solution.y))
    #print("solution.y:", solution.y)
    return solution.y[:, -1]


def get_new_lambda_u(t, lambda_matrix, u_init):
    exp_matrix = expm(lambda_matrix * t)
    current_u = np.dot(exp_matrix, u_init)
    #new_u_input = lambda_matrix * current_u_input
    return current_u

# Function to apply a fixed random control input
def apply_random_step(mjData, mjmodel, duration, scale):
    fixed_control = np.random.uniform(-scale, scale, mjmodel.nu)  # Fixed random control
    end_time = mjData.time + duration
    mjData.ctrl[:] = fixed_control
    while mjData.time < end_time:
        mujoco.mj_step(mjmodel, mjData)
    mjData.ctrl[:] = 0

# Function to reset simulation after initialization
def reset_sim(mjData, qpos, qvel):
    mjData.qpos[:] = qpos
    mjData.qvel[:] = qvel
    mjData.time = 0.0

# Function to check convergence to the origin
def has_converged(states, threshold, window_size):
    if len(states) < window_size:
        return False
    window = np.array(states[-window_size:])
    return np.all(np.abs(window) < threshold)

def get_ee_position(data, ee_site_id, steady_state_z_values):
    return data.site_xpos[ee_site_id] - np.array([0, 0, steady_state_z_values[0]])

def get_geom_positions(geom_ids_to_include, data, steady_state_z_values):
    # get current relative geoms positions
    geom_positions = []
    for geom_id in geom_ids_to_include:
        geom_positions.append(data.geom_xpos[geom_id] - np.array([0, 0, steady_state_z_values[geom_id]]))  # Append x, y, z, geom ID also works for z values as EE is at index 0
    return geom_positions

def sample_rate_with_bounds(current_control, control_scale, control_scale_rate, interval_duration, nu):

    # Calculate maximum and minimum allowable rates for each actuator
    max_rate = (control_scale - current_control) / interval_duration
    min_rate =  - current_control / interval_duration

    # Clip the rate bounds according to control_scale_rate
    max_rate = np.clip(max_rate, -control_scale_rate, control_scale_rate)
    min_rate = np.clip(min_rate, -control_scale_rate, control_scale_rate)

    # Sample a random rate within the clipped range
    rate = np.random.uniform(min_rate, max_rate, nu)
    return rate

def optimize_acceleration(current_control, current_rate, control_scale, control_scale_rate, interval_duration, nu):
    """
    Optimize the acceleration for the control input to satisfy rate and control limits.

    Parameters:
        current_control (np.ndarray): Current control input values.
        current_rate (np.ndarray): Current rate of change of control inputs (derivative of control).
        control_scale (float): The maximum allowed magnitude for control inputs.
        control_scale_rate (float): The maximum allowed magnitude for the rate of change of control inputs.
        interval_duration (float): Duration for which the acceleration is applied.
        nu (int): Number of actuators.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Optimized rate and control input after applying the acceleration.
    """
    optimized_rates = np.zeros(nu)
    optimized_controls = np.zeros(nu)

    for i in range(nu):
        # Objective function: Minimize the magnitude of the acceleration
        def objective(acc):
            return acc**2

        # Constraints for rate and control bounds
        def rate_constraint(acc):
            return control_scale_rate - abs(current_rate[i] + acc * interval_duration)

        def control_constraint(acc):
            updated_rate = current_rate[i] + acc * interval_duration
            updated_control = current_control[i] + updated_rate * interval_duration
            return control_scale - abs(updated_control)

        # Define constraints for the optimization
        constraints = [
            {"type": "ineq", "fun": rate_constraint},
            {"type": "ineq", "fun": control_constraint},
        ]

        # Solve the optimization for the current actuator
        result = minimize(
            fun=objective,
            x0=0,  # Initial guess for acceleration
            constraints=constraints,
            method='SLSQP',  # Sequential Least Squares Programming
        )

        if result.success:
            # Compute the updated rate and control
            optimized_acc = result.x[0]
            optimized_rate = current_rate[i] + optimized_acc * interval_duration
            optimized_control = current_control[i] + optimized_rate * interval_duration

            optimized_rates[i] = optimized_rate
            optimized_controls[i] = optimized_control
        else:
            raise ValueError(f"Optimization failed for actuator {i}.")

    return optimized_rates, optimized_controls
