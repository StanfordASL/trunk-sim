import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

#%matplotlib inline
import matplotlib
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams

rcParams["font.serif"] = "cmr14"
rcParams['savefig.dpi'] = 300
rcParams["figure.dpi"] = 100
rcParams.update({'font.size': 18})
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 3.5
params = {'legend.fontsize': 12,
          'legend.handlelength': 2}
plt.rcParams.update(params)

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def visualize_trajectory_from_data(df: pd.DataFrame, links: Optional[List[int]] = None):
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111, projection='3d')
    for i in links:
        i = i
        ax.plot(df[f"x{i}"], df[f"y{i}"], df[f"z{i}"], label=f"Trajectory {i}")

    # aspect ratio
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.legend()

    return fig



def visualize_velocities_from_data(df: pd.DataFrame, links: Optional[List[int]] = None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # add more space between subplots
    fig.subplots_adjust(wspace=0.3)
    for i in links:
        ax[0].plot(df["t"], df[f"vx{i}"], label=f"$v_x{i}$")
        ax[1].plot(df["t"], df[f"vy{i}"], label=f"$v_y{i}$")
        ax[2].plot(df["t"], df[f"vz{i}"], label=f"$v_z{i}$")
    
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Velocity $x$')
    ax[0].legend()
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Velocity $y$')
    ax[1].legend()
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Velocity $z$')
    ax[2].legend()
    
    return fig

def visualize_inputs_from_data(df: pd.DataFrame):
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    
    # add more space between subplots
    fig.subplots_adjust(hspace=0.3)

    for i, dim in enumerate(["x", "y"]):
        for j in range(3):
            ax[i,j].plot(df["t"], df[f"u{dim}{j+1}"], label=f"$u_{dim}{j+1}$")
            ax[i,j].set_xlabel('Time')
            ax[i,j].set_ylabel(f'Input {dim}{j+1}')
            ax[i,j].legend()

    return fig