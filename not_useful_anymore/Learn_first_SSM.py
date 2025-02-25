
from ssmlearnpy.geometry.coordinates_embedding import coordinates_embedding
import matplotlib.pyplot as plt

from ssmlearnpy import SSMLearn
import numpy as np

import pandas as pd
import os

def load_and_truncate_trajectories(num_trajectories, folder_path, start_idx):
    trajectories = []
    time_values = []

    for i in range(num_trajectories):
        file_path = os.path.join(folder_path, f"autonomous_state_traj_{i}.csv")
        # Read the CSV file
        traj_df = pd.read_csv(file_path, header=None)
        # Convert the DataFrame to a numpy array and append it to the list
        # print(traj_df.values[1:])
        # print(traj_df.values[0:])
        traj_array = traj_df.values[1:].astype(np.float32)
        #print(traj_array)
        time_values.append(traj_array[start_idx:, 0])
        trajectories.append(traj_array[start_idx:, 1:].transpose())
        #print(traj_array[:, 1:].transpose())

    return trajectories, time_values

# load trajectories
num_trajectories = 9 # Specify the number of trajectories you want to load
SSM_dim = 8 # SSM image dimension
plotting = True
#ee_steady_state_position = [0, 0, 1.89144495e-01] # steady state value of the end-effector position, SSM should pass through that
poly_degree_SSM = 5 # Degree of the SSM approximation
poly_degree_dynamics = 8
current_dir = os.path.dirname(__file__)
folder_path = os.path.join(current_dir, 'trajectories', 'data')
start_idx = 400
x, t = load_and_truncate_trajectories(num_trajectories, folder_path, start_idx)
print(f"Loaded {len(x)} trajectories.")

# generate embedding
t_y, y, opts_embedding = coordinates_embedding(t, x, imdim = SSM_dim, over_embedding = 2, shift_steps =1)

# try doing singular values decompositn and project my subspace on my own
#  singular values of data matrix
y_matrix = np.concatenate(y, axis=1)
U, s, v = np.linalg.svd(y_matrix, full_matrices = False)
SSM_basis = U[:, 0:SSM_dim]
z = [SSM_basis.T @ y_element for y_element in y]

if plotting:
    # look at singular values of data matrix
    #y_matrix = np.concatenate(y, axis=1)
    #U, s, v = np.linalg.svd(y_matrix, full_matrices = False)

    # Calculate cumulative variance explained
    singular_values_squared = s ** 2
    total_variance = np.sum(singular_values_squared)
    cumulative_variance_explained = np.cumsum(singular_values_squared) / total_variance

    # Create a 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot singular value decay
    axes[0].plot(s, marker='o')
    axes[0].set_title('Decay of Singular Values')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Singular Value')

    # Plot cumulative variance explained
    axes[1].plot(cumulative_variance_explained, marker='o')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].set_xlabel('Number of Singular Values')
    axes[1].set_ylabel('Variance Explained')
    axes[1].set_ylim(0, 1)  # Ensure the y-axis is between 0 and 1 for clarity

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


# learn SSM
ssm = SSMLearn(
    t = t_y, 
    x = y, 
    reduced_coordinates = z,
    derive_embdedding = False,
    ssm_dim = SSM_dim, 
    dynamics_type = 'flow',
)

# learn reduced coorsindates via SVD
# ssm.get_reduced_coordinates('linearchart')

# fit SSM
ssm.get_parametrization(poly_degree = poly_degree_SSM, cv=2, alpha=[10])#, alpha=[1e-4, 1e-3, 0.01, 0.1, 1.0]) # fit_intercept=True

# fit reduced dynamics
ssm.get_reduced_dynamics(poly_degree = poly_degree_dynamics, cv=2, alpha=[300])#, alpha=[1e-4, 1e-3, 0.01, 0.1, 1.0])

if plotting:
    # predict input trajectories with lower order dynamics
    traj_idx = 0
    ssm.predict(idx_trajectories=[traj_idx]) #range(n_trajs)

    # investige predicted trajectory on trainings data
    t_predict = ssm.predictions['time']
    x_predict = ssm.predictions['observables']
    e_predict = ssm.predictions['errors']

    # Extract first two elements of predicted observables
    x_predict_2d = x_predict[0][:2, :]

    # Extract first two elements of true observables
    y_true_2d = y[traj_idx][:2, :]

    # Plot predicted trajectory
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true_2d[0, 0],y_true_2d[1, 0])
    plt.plot(x_predict_2d[0, :], x_predict_2d[1, :], label="Predicted Trajectory", linestyle="--")

    # Plot true trajectory
    plt.plot(y_true_2d[0, :], y_true_2d[1, :], label="True Trajectory", linestyle="-")

    # Labels and legend
    plt.xlabel("First Observable")
    plt.ylabel("Second Observable")
    plt.title("Predicted vs True Trajectories in 2D Space")
    plt.legend()
    plt.grid()
    plt.show()


print("p")
