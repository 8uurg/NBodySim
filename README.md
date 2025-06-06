# N-Body Simulator
This is a quick N-Body simulator written using Python, Numpy and Scipy. It uses the scipy integrators / solvers to maintain accuracy while being highly performant, with the calculations being vectorized to reduce Python overhead.

## Installation
The Python environment is set up using [uv](https://docs.astral.sh/uv/). Setting up the corresponding `.venv` should be as easy as running `uv sync`.
To activate the venv, run `.venv/Scripts/activate` (Windows) or `.venv/bin/activate` (Linux, MacOS).

## Running
- `example.py` shows how to use the implementation to simulate and plot. Run `python .\nbodysim\example.py`.
- Tests and benchmarks are configured and can be ran using `pytest`.