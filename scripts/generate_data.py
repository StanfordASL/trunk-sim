import os
import argparse
from tqdm import tqdm

os.environ["MUJOCO_GL"] = "egl"  # Avoids issues with GLFWError on Linux TODO: Patrick

from trunk_sim.simulator import TrunkSimulator, get_model_path
from trunk_sim.data import TrunkData
from trunk_sim.policy import TrunkPolicy
from trunk_sim.rollout import rollout


def main(args):
    simulator = TrunkSimulator(get_model_path(args.model_type))
    data = TrunkData(states="pos_vel", links=[1, 2, 3])
    policy = TrunkPolicy(lambda _: simulator.get_random_control())

    for rollout_idx in tqdm(range(1, args.num_rollouts + 1)):
        initial_state = simulator.get_random_state()
        rollout(
            data,
            simulator,
            policy,
            initial_state=initial_state,
            duration=args.duration,
            render_video=args.render_video,
            video_filename=args.videos_dir + f"rollout_{rollout_idx}.mp4",
        )

    data.save_to_csv(args.data_filename)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="default", help="Model type to use."
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of rollouts to perform."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration of each rollout in seconds.",
    )
    parser.add_argument(
        "--render_video", action="store_true", help="Render video of the rollout."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="trunk_videos/",
        help="Directory of the rendered video.",
    )
    parser.add_argument(
        "--data_filename",
        type=str,
        default="data.csv",
        help="Filename to save the data.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
