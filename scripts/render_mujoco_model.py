import os
import argparse
import numpy as np
from tqdm import tqdm
import mujoco
import mediapy as media

from trunk_sim.simulator import TrunkSimulator
from trunk_sim.data import TrunkData
from trunk_sim.policy import HarmonicPolicy, RandomWalkPolicy, steady_state_input
from trunk_sim.rollout import rollout


def main(args):
    simulator = TrunkSimulator(
        num_segments=args.num_segments, tip_mass=args.tip_mass, radius=args.radius, length=args.length, spacing=args.spacing, tip_size=args.tip_size
    )

    if not os.path.exists(args.data_folder):
        os.makedirs(args.data_folder)

    if not os.path.exists(
        os.path.join(args.data_folder, "images")
    ):
        os.makedirs(os.path.join(args.data_folder, "images"))

    if args.init_steady_state:
        angle = np.random.uniform(0,2*np.pi)
        sign = np.random.choice([-1,1])
        simulator.set_initial_steady_state(
            steady_state_input(simulator.num_segments, amplitude=np.random.uniform(0.0,20.0), angle=angle),
            kick=steady_state_input(simulator.num_segments, amplitude=np.random.uniform(0.0,10.0), angle=angle + np.pi/2 * sign) if args.kick else None,
        )

    with mujoco.Renderer(simulator.model, width=1920, height=1080) as renderer:  # Increase resolution for better quality
        for i in range(args.num_images):
            renderer.update_scene(simulator.data)
            pixels = renderer.render()
            media.write_image(os.path.join(args.data_folder, "images", f"render_{i}.png"), pixels)
            rollout(simulator, duration=0.05)


def parse_args():
    parser = argparse.ArgumentParser()

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
        "--kick",
        action="store_true",
        help="Apply a kick to the trunk after reaching a steady-state.",
    )
    parser.add_argument(
        "--num_segments", type=int, default=3, help="Number of segments in the trunk"
    )
    parser.add_argument(
        "--radius", type=float, default=0.005, help="Radius of each trunk segment"
    )
    parser.add_argument(
        "--length", type=float, default=0.32, help="Length of the trunk"
    )
    parser.add_argument(
        "--spacing", type=float, default=2.0, help="Length of the trunk"
    )
    parser.add_argument(
        "--num_images", type=int, default=10, help="Number of images to render"
    )
    parser.add_argument(
        "--tip_size", type=float, default=None, help="Size of the trunk tip"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
