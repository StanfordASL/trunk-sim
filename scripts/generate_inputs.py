import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

from trunk_sim.simulator import TrunkSimulator
from trunk_sim.data import TrunkData
from trunk_sim.policy import HarmonicPolicy, RandomWalkPolicy, steady_state_input
from trunk_sim.rollout import rollout

input_mapping_from_sim_to_real = [
    ["u2","u4"],
    ["u6","u1"],
    ["u3","u5"],
]

limits = {
    "u1": [0.0, 50.0],
    "u2": [0.0, 80.0],
    "u3": [0.0, 30.0],
    "u4": [0.0, 80.0],
    "u5": [0.0, 30.0],
    "u6": [0.0, 50.0],
}

def validate_input_limits(key, u):
    assert key in limits, f"Key {key} not in limits dictionary"
    assert limits[key][0] <= abs(u) <= limits[key][1], f"Input {key} out of bounds: {u}"

class InputData:
    def __init__(self):
        self.data = []
        self.id = 0

    def add_data(self, t, u, rollout_idx):
        data_entry = {
            "ID": self.id,
            #"time": t,
            #"rollout_idx": rollout_idx
        }

        for i in range(len(u)):
            for j in range(len(u[i])):
                key = input_mapping_from_sim_to_real[i][j]
                validate_input_limits(key, u[i][j])
                data_entry[key] = u[i][j]

        # sort keys
        data_entry = {k: data_entry[k] for k in sorted(data_entry.keys())}

        self.data.append(data_entry)
        self.id += 1

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)

    def plot_data(self):
        import matplotlib.pyplot as plt

        for key in limits.keys():
            plt.figure()
            plt.title(key)
            plt.plot([d[key] for d in self.data])
            plt.xlabel("Time")
            plt.ylabel(key)
            plt.show()
        

def main(args):
    data = InputData()

    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)

    for rollout_idx in tqdm(range(args.num_rollouts)):
        if args.policy == "harmonic":
            policy = HarmonicPolicy(
                frequency_range=[0.0,0.5], 
                amplitude_range=[
                    [0.0,80.0],
                    [0.0,50.0],
                    [0.0,30.0]
                ],
                phase_range=[0.0,2*np.pi], num_segments=args.num_segments
            )
        elif args.policy == "random_walk":
            policy = RandomWalkPolicy()
        else:
            raise ValueError(f"Invalid policy: {args.policy}. Only open-loop policies can be used.")
        
        t = 0
        for k in range(int(args.duration/args.timestep) + int(args.resting_duration/args.timestep)):
            if t < args.duration:
                control_input = policy(t, None)
            else:
                control_input = np.zeros((args.num_segments, 2))

            t += args.timestep
            data.add_data(t, control_input, rollout_idx)

    #data.plot_data()
    data.save_to_csv(os.path.join(args.data_folder, "data.csv"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_rollouts", type=int, default=100, help="Number of rollouts to perform."
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.01,
        help="Timestep of the rollout in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Duration of each rollout in seconds.",
    )
    parser.add_argument(
        "--resting_duration",
        type=float,
        default=3.0,
        help="Duration of each resting period between rollouts, in seconds.",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="trunk_data/",
        help="Directory of the rendered video.",
    )
    parser.add_argument(
        "--num_segments", type=int, default=3, help="Number of segments in the trunk"
    )
    parser.add_argument(
        "--num_links_per_segment",
        type=int,
        default=1,
        help="Number of links per segment in the trunk",
    )
    parser.add_argument(
        "--policy", type=str, default="harmonic", help="Control input to use. Options: none, harmonic, random_walk"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
