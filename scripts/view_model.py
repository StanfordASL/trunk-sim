import argparse
import os

os.environ["MUJOCO_GL"] = "egl"  # Avoids issues with GLFWError on Linux TODO: Patrick
from trunk_sim.simulator import TrunkSimulator, get_model_path, render_simulator

def main(args):
    simulator = TrunkSimulator(get_model_path(args.model_type))
    render_simulator(simulator)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="default", help="Model type to use."
    )
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())