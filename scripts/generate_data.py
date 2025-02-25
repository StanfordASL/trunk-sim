import argparse
import os

os.environ["MUJOCO_GL"] = "egl"  # Avoids issues with GBM on Linux TODO: Patrick

from trunk_sim.simulator import TrunkSimulator, get_model_path
from trunk_sim.rollout import rollout
from trunk_sim import data


def main(args):
    simulator = TrunkSimulator(get_model_path(args.model_type))
    # data = ...
    # policy = ...

    rollout(
        simulator,
        num_rollouts=args.num_rollouts,
        duration_s=args.duration_s,
        timestep_ms=args.timestep_ms,
        render_video=args.render_video,
        video_filename=args.video_filename,
    )

    # Save data
    # data.save("data.pkl")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="default", help="Model type to use."
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=1, help="Number of rollouts to perform."
    )
    parser.add_argument(
        "--duration_s",
        type=float,
        default=1.0,
        help="Duration of the rollout in seconds.",
    )
    parser.add_argument(
        "--timestep_ms",
        type=float,
        default=10,
        help="Timestep of the rollout in milliseconds.",
    )
    parser.add_argument(
        "--render_video", action="store_true", help="Render video of the rollout."
    )
    parser.add_argument(
        "--video_filename",
        type=str,
        default="render.mpy",
        help="Filename of the rendered video.",
    )
    #TODO: Hugo add data_filename and policy
    # parser.add_argument(
    #     "--data_filename",
    #     type=str,
    #     default="data.csv",
    #     help="Filename to save the data.",
    # )
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())