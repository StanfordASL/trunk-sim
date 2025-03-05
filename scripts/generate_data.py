import os
import argparse
import numpy as np
from tqdm import tqdm


from trunk_sim.simulator import TrunkSimulator
from trunk_sim.data import TrunkData
from trunk_sim.policy import TrunkPolicy, HarmonicPolicy, RandomWalkPolicy, steady_state_input
from trunk_sim.rollout import rollout


def main(args):
    simulator = TrunkSimulator(
        num_segments=args.num_segments, tip_mass=args.tip_mass
    )
    
    data = TrunkData(
        simulator.num_links_per_segment,
        simulator.num_segments,
        states="pos_vel",
        links="all",
    )

    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)

    if args.render_video and not os.path.exists(
        os.path.join(args.data_folder, "videos")
    ):
        os.makedirs(os.path.join(args.data_folder, "videos"))

    for rollout_idx in tqdm(range(1, args.num_rollouts + 1)):
        if args.policy == "harmonic":
            policy = HarmonicPolicy(
                frequency=np.random.uniform(0.01, 1.0), amplitude=np.random.uniform(0.01,2.0), phase=np.random.uniform(0,2*np.pi), num_segments=simulator.num_segments
            )
        elif args.policy == "random_walk":
            policy = RandomWalkPolicy()
        elif args.policy == "none":
            policy = None
        else:
            raise ValueError(f"Invalid policy: {args.policy}")

        if args.init_steady_state:
            angle = np.random.uniform(0,2*np.pi)
            sign = np.random.choice([-1,1])
            simulator.set_initial_steady_state(
                steady_state_input(simulator.num_segments, amplitude=np.random.uniform(0.0,12.0), angle=angle),
                kick=steady_state_input(simulator.num_segments, amplitude=np.random.uniform(0.0,10.0), angle=angle + np.pi/2 * sign),
            )

        rollout(
            simulator=simulator,
            policy=policy,
            data=data,
            duration=args.duration,
            render_video=args.render_video,
            video_filename=os.path.join(
                args.data_folder, "videos", f"rollout_{rollout_idx}.mp4"
            ),
            stop_at_convergence=True
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
        default=5.0,
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
        "--init_steady_state",
        action="store_true",
        help="Initialize the trunk in a steady state configuration.",
    )
    parser.add_argument(
        "--num_segments", type=int, default=3, help="Number of segments in the trunk"
    )
    parser.add_argument(
        "--policy", type=str, default="none", help="Control input to use. Options: none, harmonic, random_walk"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
