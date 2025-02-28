# trunk-sim

The ASL Trunk simulator is a tool designed to simulate the dynamics of a trunk system. It provides functionalities to generate, manipulate, and visualize trunk models, as well as to run simulations and analyze the results based on a user-defined control policy.

## Features

- Generate trunk models with different properties
- Add and manipulate simulation data
- Create PyTorch datasets for machine learning
- Visualize and evaluate policies in simulation

## Usage

### Generating Data

To generate simulation data, run:

```bash
uv run scripts/generate_data.py --render_video
```

### Running Tests

Run tests with:

```bash
pytest
```

## Acknowledgements

This project is developed and maintained by the ASL team at Stanford University.
