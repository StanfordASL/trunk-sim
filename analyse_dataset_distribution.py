import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def analyze_datasets(dataset_dir, truncation_index):
    """
    Analyzes CSV datasets in a given directory by plotting:
      - A 2D heat map for x_ee vs y_ee (with log normalization).
      - A 1D histogram for z_ee on a log scale (raw counts).

    Only rows after the specified truncation index in each file are considered.
    """
    x_vals_list = []
    y_vals_list = []
    z_vals_list = []

    # Read CSV files, truncate rows, and collect data.
    for file in os.listdir(dataset_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(dataset_dir, file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # Only consider rows after the truncation index.
            df_trunc = df.iloc[truncation_index:]

            # Ensure columns exist.
            for col in ["x_ee", "y_ee", "z_ee"]:
                if col not in df_trunc.columns:
                    raise ValueError(f"Column {col} not found in file {file_path}.")

            x_vals_list.append(df_trunc["x_ee"].values)
            y_vals_list.append(df_trunc["y_ee"].values)
            z_vals_list.append(df_trunc["z_ee"].values)

    if not (x_vals_list and y_vals_list and z_vals_list):
        print("No data found. Please check the dataset directory and truncation index.")
        return

    # Concatenate data across files.
    x_vals = np.concatenate(x_vals_list)
    y_vals = np.concatenate(y_vals_list)
    z_vals = np.concatenate(z_vals_list)

    # ---------------------------
    # Figure 1: 2D Heatmap (x_ee vs y_ee)
    # ---------------------------
    fig1 = plt.figure(figsize=(8, 6))
    plt.hist2d(x_vals, y_vals, bins=50, cmap='hot', norm=LogNorm())
    plt.colorbar(label='Counts (log scale)')
    plt.xlabel('x_ee')
    plt.ylabel('y_ee')
    plt.title('xy distribution')
    plt.tight_layout()

    # ---------------------------
    # Figure 2: 1D Histogram (z_ee, log scale on y-axis)
    # ---------------------------
    fig2 = plt.figure(figsize=(8, 6))
    plt.hist(z_vals, bins=50, color='blue', alpha=0.7, log=True)
    plt.xlabel('z_ee')
    plt.ylabel('Number of Counts (log scale)')
    plt.title('z distribution')
    plt.tight_layout()

    # Show both figures at the same time.
    plt.show()

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join("..", "larger_sets", "data_even_larger2"))
    truncation_index = 300  # Only rows after this index will be analyzed.
    analyze_datasets(dataset_dir, truncation_index)
