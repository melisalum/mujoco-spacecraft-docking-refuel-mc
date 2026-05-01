# MuJoCo Spacecraft Docking & Refuelling Monte Carlo Simulation

Dual-mode Python simulation for orbital proximity operations using MuJoCo:

- `docking` mode: chaser approaches a fixed target body.
- `refuel` mode: probe tip aligns with a fixed refuelling port.

The script is designed for fast Monte Carlo experimentation with simple rigid-body models, bounded PD control, and CSV/plot outputs.

## Project Goals

- Keep the environment lightweight and reproducible:
  - Inline MuJoCo XML (no external assets, no meshes).
  - Primitive geoms only.
- Support repeated trials for robustness analysis.
- Provide configurable control and success criteria from CLI.
- Include runtime safeguards for MuJoCo import/startup issues.

## Core Features

- Single-file implementation: `mujoco_docking.py`
- Two simulation modes via CLI:
  - `--mode docking` (default)
  - `--mode refuel`
- Bounded PD translational force controller.
- Optional Gaussian sensor noise in refuel mode.
- Monte Carlo batch execution with:
  - Success/failure statistics
  - Time-to-target statistics
  - Per-trial CSV export
  - Histogram + scatter diagnostic plots
- Preflight system:
  - Import-only MuJoCo check in a child process
  - One-shot smoke simulation in a child process
  - Timeout guards and optional verbose logging

## Simulation Modes

### 1) Docking Mode

- Bodies:
  - Fixed target box
  - Free-joint chaser box
- Success condition:
  - Distance to target < `--dock-distance-threshold`
  - Speed < `--dock-speed-threshold`
- Default outputs:
  - `mc_docking_results.csv`
  - `mc_time_to_dock_hist.png`
  - `mc_initial_distance_vs_success.png`

### 2) Refuel Mode

- Bodies:
  - Fixed refuelling port (cylinder-based ring-like approximation)
  - Free-joint probe with shaft + tip site
- Controller target:
  - Align probe tip to port center
- Measurement model:
  - `measured_error = true_error + N(0, sensor_noise_std^2)`
- Success condition:
  - Alignment error < `--alignment-threshold`
  - Speed < `--speed-threshold`
- Default outputs:
  - `mc_refuel_results.csv`
  - `mc_refuel_time_to_align_hist.png`
  - `mc_refuel_initial_error_vs_success.png`

## Requirements

- Python 3.10+
- `numpy`
- `mujoco` (Python bindings)
- `matplotlib`

Install:

```bash
pip install numpy mujoco matplotlib
```

## Usage

### Basic Runs

Docking (default mode):

```bash
python3 mujoco_docking.py
```

Refuel mode:

```bash
python3 mujoco_docking.py --mode refuel
```

### Fast Sanity Runs

```bash
python3 mujoco_docking.py --trials 5 --sim-duration 1.0 --no-plots
python3 mujoco_docking.py --mode refuel --trials 5 --sim-duration 1.0 --no-plots
```

### Useful Options

- `--trials`: number of Monte Carlo trials
- `--seed`: random seed
- `--sim-duration`: per-trial simulation horizon (s)
- `--kp`, `--kd`, `--max-force`: PD parameters and force bound
- `--no-plots`: skip histogram/scatter generation
- `--csv-output`, `--hist-output`, `--scatter-output`: override output paths

Mode-specific:

- Docking:
  - `--dock-distance-threshold`
  - `--dock-speed-threshold`
- Refuel:
  - `--sensor-noise-std`
  - `--alignment-threshold`
  - `--speed-threshold`

Preflight/troubleshooting:

- `--skip-preflight`
- `--import-timeout`
- `--preflight-timeout`
- `--verbose-preflight`

## Preflight Strategy

Before full Monte Carlo execution (unless `--skip-preflight`):

1. Import-check child process verifies `import mujoco`.
2. Mode-specific one-shot simulation smoke test runs.
3. Parent process enforces timeouts and reports failures clearly.

This prevents long hangs from blocking full experiment runs.

## Output Data

CSV files contain one row per trial with:

- Trial metadata (`trial_id`, `mode`, initial position)
- Success label
- Time metric (`time_to_dock` or `time_to_align`)
- Final and minimum error metrics
- Final speed
- Refuel noise setting (`sensor_noise_std`) in refuel mode

## Notes

- No rendering is included by design.
- Models are intentionally minimal for rapid iteration and controller experiments.
- The architecture is ready to extend into larger batch studies, controller ablations, and parameter sweeps.
