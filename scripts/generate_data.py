import os
import argparse
import numpy as np
from tqdm import tqdm


from trunk_sim.simulator import TrunkSimulator
from trunk_sim.data import TrunkData
from trunk_sim.policy import TrunkPolicy, HarmonicPolicy
from trunk_sim.rollout import rollout


def main(args):
    simulator = TrunkSimulator(num_segments=1, num_links_per_segment=10, tip_mass=args.tip_mass)
    data = TrunkData(simulator.num_links_per_segment, simulator.num_segments, states="pos_vel", segments="all")
    policy = None #HarmonicPolicy(frequency=1.0, amplitude=1.0, phase=0.0, num_segments=simulator.num_segments)

    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)

    if args.render_video and not os.path.exists(os.path.join(args.data_folder, "videos")):
        os.makedirs(os.path.join(args.data_folder, "videos"))

    for rollout_idx in tqdm(range(1, args.num_rollouts + 1)):

        simulator.set_initial_steady_state(steady_state_control_input=np.ones((simulator.num_segments, simulator.num_controls_per_segment)), max_duration=10)

        rollout(
            simulator=simulator,
            policy=policy,
            data=data,
            duration=args.duration,
            render_video=args.render_video,
            video_filename=os.path.join(args.data_folder, "videos", f"rollout_{rollout_idx}.mp4")
        )

    data.save_to_csv(os.path.join(args.data_folder, "data.csv"))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of rollouts to perform."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of each rollout in seconds.",
    )
    parser.add_argument(
        "--render_video", action="store_true", help="Render video of the rollout."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="trunk_data/",
        help="Directory of the rendered video.",
    )
    parser.add_argument(
        "--tip_mass",
        type=float,
        default=0.1,
        help="Mass of the trunk tip.",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=3,
        help="Number of segments in the trunk"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
