#!/usr/bin/env python3
"""Dual-mode MuJoCo Monte Carlo script: docking and refuelling alignment."""

import argparse
import csv
import subprocess
import sys

import numpy as np


# ---------------------------------------------------------------------------
# 1) Inline model definitions (XML strings)
# ---------------------------------------------------------------------------
DOCKING_MODEL_XML = """
<mujoco model="spacecraft_docking">
  <option timestep="0.01" gravity="0 0 0" integrator="RK4"/>

  <worldbody>
    <!-- Fixed target spacecraft -->
    <body name="target" pos="0 0 0">
      <geom name="target_geom" type="box" size="0.1 0.1 0.1" mass="5.0" rgba="0.9 0.3 0.3 1"/>
    </body>

    <!-- Free-floating chaser spacecraft -->
    <body name="chaser" pos="-1.0 0.25 0.0">
      <freejoint name="chaser_free"/>
      <geom name="chaser_geom" type="box" size="0.08 0.08 0.08" mass="2.0" rgba="0.2 0.4 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

REFUEL_MODEL_XML = """
<mujoco model="spacecraft_refuel">
  <option timestep="0.01" gravity="0 0 0" integrator="RK4"/>

  <worldbody>
    <!-- Fixed refuelling port (ring-like approximation) -->
    <body name="refuel_port" pos="0 0 0">
      <geom name="port_outer" type="cylinder" size="0.11 0.012" rgba="0.80 0.80 0.80 1"/>
      <geom name="port_inner" type="cylinder" size="0.06 0.013" rgba="0.15 0.15 0.15 1"/>
      <site name="port_center" pos="0 0 0" size="0.005" rgba="0.0 1.0 0.0 1"/>
    </body>

    <!-- Free-floating probe -->
    <body name="probe" pos="-1.0 0.2 0.0">
      <freejoint name="probe_free"/>
      <geom name="probe_shaft" type="capsule" fromto="-0.10 0 0 0.10 0 0" size="0.02" mass="1.5" rgba="0.2 0.5 0.9 1"/>
      <geom name="probe_tip_geom" type="sphere" pos="0.10 0 0" size="0.024" rgba="0.95 0.85 0.2 1"/>
      <site name="probe_tip" pos="0.10 0 0" size="0.004" rgba="1.0 1.0 0.0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# 2) Defaults and runtime settings (CLI can override)
# ---------------------------------------------------------------------------
DEFAULT_MODE = "docking"
DEFAULT_SIM_DURATION_S = 7.0
DEFAULT_N_TRIALS = 100
DEFAULT_RNG_SEED = 7

DEFAULT_KP = 12.0
DEFAULT_KD = 6.0
DEFAULT_MAX_FORCE = 25.0

DEFAULT_DOCKING_DISTANCE_THRESHOLD = 0.15
DEFAULT_DOCKING_SPEED_THRESHOLD = 0.05

DEFAULT_ALIGNMENT_THRESHOLD = 0.05
DEFAULT_REFUEL_SPEED_THRESHOLD = 0.05
DEFAULT_SENSOR_NOISE_STD = 0.0

DEFAULT_DOCKING_CSV_OUTPUT = "mc_docking_results.csv"
DEFAULT_DOCKING_HIST_OUTPUT = "mc_time_to_dock_hist.png"
DEFAULT_DOCKING_SCATTER_OUTPUT = "mc_initial_distance_vs_success.png"

DEFAULT_REFUEL_CSV_OUTPUT = "mc_refuel_results.csv"
DEFAULT_REFUEL_HIST_OUTPUT = "mc_refuel_time_to_align_hist.png"
DEFAULT_REFUEL_SCATTER_OUTPUT = "mc_refuel_initial_error_vs_success.png"

DEFAULT_IMPORT_TIMEOUT_S = 8.0
DEFAULT_PREFLIGHT_TIMEOUT_S = 15.0


# Runtime configuration consumed by run functions.
SIM_DURATION_S = DEFAULT_SIM_DURATION_S
KP = DEFAULT_KP
KD = DEFAULT_KD
MAX_FORCE = DEFAULT_MAX_FORCE
DOCKING_DISTANCE_THRESHOLD = DEFAULT_DOCKING_DISTANCE_THRESHOLD
DOCKING_SPEED_THRESHOLD = DEFAULT_DOCKING_SPEED_THRESHOLD
ALIGNMENT_THRESHOLD = DEFAULT_ALIGNMENT_THRESHOLD
REFUEL_SPEED_THRESHOLD = DEFAULT_REFUEL_SPEED_THRESHOLD


_MUJOCO = None
_NOISE_RNG = np.random.default_rng(DEFAULT_RNG_SEED)


def get_mujoco():
    """Lazily import MuJoCo so subprocess preflight can guard import hangs."""
    global _MUJOCO
    if _MUJOCO is None:
        import mujoco  # Local import by design.

        _MUJOCO = mujoco
    return _MUJOCO


def pd_force_from_error(
    position_error: np.ndarray,
    current_velocity: np.ndarray,
    target_velocity: np.ndarray | None = None,
    kp: float | None = None,
    kd: float | None = None,
    max_force: float | None = None,
) -> np.ndarray:
    """Compute bounded translational force from PD terms."""
    if kp is None:
        kp = KP
    if kd is None:
        kd = KD
    if max_force is None:
        max_force = MAX_FORCE
    if target_velocity is None:
        target_velocity = np.zeros(3)

    vel_error = target_velocity - current_velocity
    force = kp * position_error + kd * vel_error

    force_norm = np.linalg.norm(force)
    if force_norm > max_force:
        force *= max_force / force_norm
    return force


# ---------------------------------------------------------------------------
# 3) Single-trial simulation functions
# ---------------------------------------------------------------------------
def run_one_sim(initial_position, trial_id=None):
    """Run one docking trial and return scalar metrics."""
    mujoco = get_mujoco()
    model = mujoco.MjModel.from_xml_string(DOCKING_MODEL_XML)
    data = mujoco.MjData(model)

    chaser_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chaser")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    chaser_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chaser_free")

    chaser_qpos_adr = model.jnt_qposadr[chaser_joint_id]
    chaser_dof_adr = model.jnt_dofadr[chaser_joint_id]

    initial_position = np.asarray(initial_position, dtype=float)
    data.qpos[chaser_qpos_adr : chaser_qpos_adr + 3] = initial_position
    data.qpos[chaser_qpos_adr + 3 : chaser_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[chaser_dof_adr : chaser_dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    target_pos = data.xpos[target_body_id].copy()
    initial_distance = float(np.linalg.norm(target_pos - data.xpos[chaser_body_id]))

    steps = int(SIM_DURATION_S / model.opt.timestep)
    success = False
    time_to_dock = None
    distance_log = []
    speed_log = []

    for _ in range(steps):
        chaser_pos = data.xpos[chaser_body_id].copy()
        chaser_vel = data.qvel[chaser_dof_adr : chaser_dof_adr + 3].copy()

        position_error = target_pos - chaser_pos
        force_cmd = pd_force_from_error(position_error, chaser_vel)
        data.xfrc_applied[chaser_body_id, :3] = force_cmd
        data.xfrc_applied[chaser_body_id, 3:] = 0.0

        distance = float(np.linalg.norm(position_error))
        speed = float(np.linalg.norm(chaser_vel))
        distance_log.append(distance)
        speed_log.append(speed)

        if distance < DOCKING_DISTANCE_THRESHOLD and speed < DOCKING_SPEED_THRESHOLD:
            success = True
            time_to_dock = float(data.time)
            break

        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    final_pos = data.xpos[chaser_body_id].copy()
    final_vel = data.qvel[chaser_dof_adr : chaser_dof_adr + 3].copy()

    final_distance = float(np.linalg.norm(target_pos - final_pos))
    final_speed = float(np.linalg.norm(final_vel))
    min_distance = float(min(min(distance_log), final_distance)) if distance_log else final_distance

    result = {
        "mode": "docking",
        "success": success,
        "time_to_dock": time_to_dock,
        "final_distance": final_distance,
        "final_speed": final_speed,
        "min_distance": min_distance,
        "initial_distance": initial_distance,
    }
    if trial_id is not None:
        result["trial_id"] = trial_id
    return result


def run_one_refuel_sim(initial_position, sensor_noise_std=0.0, trial_id=None):
    """Run one refuelling alignment trial and return scalar metrics."""
    mujoco = get_mujoco()
    model = mujoco.MjModel.from_xml_string(REFUEL_MODEL_XML)
    data = mujoco.MjData(model)

    probe_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
    probe_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_free")
    probe_tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "probe_tip")
    port_center_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "port_center")

    probe_qpos_adr = model.jnt_qposadr[probe_joint_id]
    probe_dof_adr = model.jnt_dofadr[probe_joint_id]

    initial_position = np.asarray(initial_position, dtype=float)
    data.qpos[probe_qpos_adr : probe_qpos_adr + 3] = initial_position
    data.qpos[probe_qpos_adr + 3 : probe_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[probe_dof_adr : probe_dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)

    port_center_pos = data.site_xpos[port_center_site_id].copy()
    probe_tip_pos = data.site_xpos[probe_tip_site_id].copy()
    initial_alignment_error = float(np.linalg.norm(port_center_pos - probe_tip_pos))

    steps = int(SIM_DURATION_S / model.opt.timestep)
    success = False
    time_to_align = None
    alignment_error_log = []
    speed_log = []

    for _ in range(steps):
        probe_tip_pos = data.site_xpos[probe_tip_site_id].copy()
        probe_vel = data.qvel[probe_dof_adr : probe_dof_adr + 3].copy()

        true_error = port_center_pos - probe_tip_pos
        if sensor_noise_std > 0.0:
            measured_error = true_error + _NOISE_RNG.normal(0.0, sensor_noise_std, size=3)
        else:
            measured_error = true_error

        force_cmd = pd_force_from_error(measured_error, probe_vel)
        data.xfrc_applied[probe_body_id, :3] = force_cmd
        data.xfrc_applied[probe_body_id, 3:] = 0.0

        alignment_error = float(np.linalg.norm(true_error))
        speed = float(np.linalg.norm(probe_vel))
        alignment_error_log.append(alignment_error)
        speed_log.append(speed)

        if alignment_error < ALIGNMENT_THRESHOLD and speed < REFUEL_SPEED_THRESHOLD:
            success = True
            time_to_align = float(data.time)
            break

        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    final_tip_pos = data.site_xpos[probe_tip_site_id].copy()
    final_probe_vel = data.qvel[probe_dof_adr : probe_dof_adr + 3].copy()

    final_alignment_error = float(np.linalg.norm(port_center_pos - final_tip_pos))
    final_speed = float(np.linalg.norm(final_probe_vel))
    min_alignment_error = (
        float(min(min(alignment_error_log), final_alignment_error))
        if alignment_error_log
        else final_alignment_error
    )

    result = {
        "mode": "refuel",
        "success": success,
        "time_to_align": time_to_align,
        "final_alignment_error": final_alignment_error,
        "final_speed": final_speed,
        "min_alignment_error": min_alignment_error,
        "initial_alignment_error": initial_alignment_error,
        "sensor_noise_std": float(sensor_noise_std),
    }
    if trial_id is not None:
        result["trial_id"] = trial_id
    return result


# ---------------------------------------------------------------------------
# 4) Monte Carlo utility helpers
# ---------------------------------------------------------------------------
def sample_initial_position(rng, mode):
    """Sample random initial probe/chaser position for the selected mode."""
    if mode == "docking":
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.4, 1.8)
        return direction * radius

    # Refuel mode: start generally "in front of" the port with lateral offsets.
    x = rng.uniform(-1.4, -0.4)
    y = rng.uniform(-0.5, 0.5)
    z = rng.uniform(-0.5, 0.5)
    return np.array([x, y, z], dtype=float)


def resolve_output_paths(args):
    """Resolve output paths with mode-specific defaults."""
    if args.mode == "docking":
        csv_output = args.csv_output or DEFAULT_DOCKING_CSV_OUTPUT
        hist_output = args.hist_output or DEFAULT_DOCKING_HIST_OUTPUT
        scatter_output = args.scatter_output or DEFAULT_DOCKING_SCATTER_OUTPUT
    else:
        csv_output = args.csv_output or DEFAULT_REFUEL_CSV_OUTPUT
        hist_output = args.hist_output or DEFAULT_REFUEL_HIST_OUTPUT
        scatter_output = args.scatter_output or DEFAULT_REFUEL_SCATTER_OUTPUT
    return csv_output, hist_output, scatter_output


def save_results_csv(results, initial_positions, csv_path, mode):
    """Save per-trial scalar metrics to CSV."""
    if mode == "docking":
        fieldnames = [
            "trial_id",
            "mode",
            "success",
            "time_to_dock",
            "initial_distance",
            "final_distance",
            "final_speed",
            "min_distance",
            "init_x",
            "init_y",
            "init_z",
        ]
    else:
        fieldnames = [
            "trial_id",
            "mode",
            "success",
            "time_to_align",
            "initial_alignment_error",
            "final_alignment_error",
            "final_speed",
            "min_alignment_error",
            "sensor_noise_std",
            "init_x",
            "init_y",
            "init_z",
        ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (result, init_pos) in enumerate(zip(results, initial_positions), start=1):
            row = {
                "trial_id": idx,
                "mode": result.get("mode", mode),
                "success": int(result["success"]),
                "init_x": float(init_pos[0]),
                "init_y": float(init_pos[1]),
                "init_z": float(init_pos[2]),
            }
            if mode == "docking":
                row.update(
                    {
                        "time_to_dock": "" if result["time_to_dock"] is None else result["time_to_dock"],
                        "initial_distance": result["initial_distance"],
                        "final_distance": result["final_distance"],
                        "final_speed": result["final_speed"],
                        "min_distance": result["min_distance"],
                    }
                )
            else:
                row.update(
                    {
                        "time_to_align": "" if result["time_to_align"] is None else result["time_to_align"],
                        "initial_alignment_error": result["initial_alignment_error"],
                        "final_alignment_error": result["final_alignment_error"],
                        "final_speed": result["final_speed"],
                        "min_alignment_error": result["min_alignment_error"],
                        "sensor_noise_std": result["sensor_noise_std"],
                    }
                )
            writer.writerow(row)


def plot_time_histogram(values, output_path, x_label, title):
    """Save histogram for successful time metrics."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    if values.size > 0:
        bins = min(15, values.size)
        plt.hist(values, bins=bins, color="tab:blue", edgecolor="black", alpha=0.85)
        plt.xlabel(x_label)
        plt.ylabel("Count")
        plt.title(title)
    else:
        plt.text(0.5, 0.5, "No successful trials", ha="center", va="center", transform=plt.gca().transAxes)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_initial_metric_vs_success(initial_metric, success_flags, output_path, x_label, title):
    """Save scatter plot of initial metric vs success."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.scatter(initial_metric, success_flags, alpha=0.8, color="tab:green")
    plt.xlabel(x_label)
    plt.ylabel("Success (0 or 1)")
    plt.title(title)
    plt.yticks([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def configure_runtime(args):
    """Apply CLI values to module-level runtime settings."""
    global SIM_DURATION_S, KP, KD, MAX_FORCE
    global DOCKING_DISTANCE_THRESHOLD, DOCKING_SPEED_THRESHOLD
    global ALIGNMENT_THRESHOLD, REFUEL_SPEED_THRESHOLD
    global _NOISE_RNG

    SIM_DURATION_S = args.sim_duration
    KP = args.kp
    KD = args.kd
    MAX_FORCE = args.max_force

    DOCKING_DISTANCE_THRESHOLD = args.dock_distance_threshold
    DOCKING_SPEED_THRESHOLD = args.dock_speed_threshold
    ALIGNMENT_THRESHOLD = args.alignment_threshold
    REFUEL_SPEED_THRESHOLD = args.speed_threshold

    # Separate RNG stream for measurement noise.
    _NOISE_RNG = np.random.default_rng(args.seed + 1000)


# ---------------------------------------------------------------------------
# 5) Preflight and import checks (preserved structure, now mode-aware)
# ---------------------------------------------------------------------------
def run_preflight_child_once(args):
    """Execute one local MuJoCo smoke run inside a child process."""
    if args.mode == "docking":
        preflight_initial_position = np.array([0.8, -0.3, 0.2], dtype=float)
        preflight_result = run_one_sim(preflight_initial_position, trial_id="preflight-child")
        print(
            "[preflight:docking] one-shot run completed "
            f"(success={preflight_result['success']}, "
            f"final_distance={preflight_result['final_distance']:.4f}, "
            f"final_speed={preflight_result['final_speed']:.4f})"
        )
    else:
        preflight_initial_position = np.array([-0.8, 0.15, -0.1], dtype=float)
        preflight_result = run_one_refuel_sim(
            preflight_initial_position,
            sensor_noise_std=args.sensor_noise_std,
            trial_id="preflight-child",
        )
        print(
            "[preflight:refuel] one-shot run completed "
            f"(success={preflight_result['success']}, "
            f"final_alignment_error={preflight_result['final_alignment_error']:.4f}, "
            f"final_speed={preflight_result['final_speed']:.4f})"
        )


def run_preflight_smoke_test(args):
    """Run mode-aware preflight in a subprocess with a hard timeout."""
    cmd = [
        sys.executable,
        sys.argv[0],
        "--internal-preflight-child",
        "--mode",
        args.mode,
        "--sim-duration",
        str(args.sim_duration),
        "--kp",
        str(args.kp),
        "--kd",
        str(args.kd),
        "--max-force",
        str(args.max_force),
        "--dock-distance-threshold",
        str(args.dock_distance_threshold),
        "--dock-speed-threshold",
        str(args.dock_speed_threshold),
        "--alignment-threshold",
        str(args.alignment_threshold),
        "--speed-threshold",
        str(args.speed_threshold),
        "--sensor-noise-std",
        str(args.sensor_noise_std),
        "--seed",
        str(args.seed),
    ]
    run_guarded_child_process(
        cmd=cmd,
        timeout_s=args.preflight_timeout,
        label=f"preflight-{args.mode}",
        verbose=args.verbose_preflight,
    )


def run_import_check_child_once():
    """Import MuJoCo once and exit; used by timeout-guarded parent preflight."""
    _ = get_mujoco()
    print("[import-check] MuJoCo import succeeded")


def run_import_check(args):
    """Run import-only check in a subprocess with a hard timeout."""
    cmd = [
        sys.executable,
        sys.argv[0],
        "--internal-import-check-child",
    ]
    run_guarded_child_process(
        cmd=cmd,
        timeout_s=args.import_timeout,
        label="import-check",
        verbose=args.verbose_preflight,
    )


def run_guarded_child_process(cmd, timeout_s, label, verbose=False):
    """Run a child process with timeout and optional verbose log output."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_text, stderr_text = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        stdout_text, stderr_text = process.communicate()
        detail = ""
        if verbose:
            detail = f" | stdout: {stdout_text.strip()} | stderr: {stderr_text.strip()}"
        raise RuntimeError(f"MuJoCo {label} timed out after {timeout_s:.2f}s.{detail}") from exc

    if verbose and stdout_text.strip():
        print(stdout_text.strip())
    if verbose and stderr_text.strip():
        print(stderr_text.strip())

    if process.returncode != 0:
        detail_parts = []
        if stdout_text.strip():
            detail_parts.append(f"stdout: {stdout_text.strip()}")
        if stderr_text.strip():
            detail_parts.append(f"stderr: {stderr_text.strip()}")
        detail = "" if not detail_parts else " | " + " | ".join(detail_parts)
        raise RuntimeError(f"MuJoCo {label} failed with exit code {process.returncode}.{detail}")


# ---------------------------------------------------------------------------
# 6) CLI parsing and main Monte Carlo loops
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-mode MuJoCo Monte Carlo simulation (docking or refuel)."
    )
    parser.add_argument(
        "--mode",
        choices=["docking", "refuel"],
        default=DEFAULT_MODE,
        help="Simulation mode: docking or refuel.",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_N_TRIALS, help="Number of Monte Carlo trials.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RNG_SEED, help="Random seed.")
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=DEFAULT_SIM_DURATION_S,
        help="Simulation duration per trial (seconds).",
    )
    parser.add_argument("--kp", type=float, default=DEFAULT_KP, help="PD proportional gain.")
    parser.add_argument("--kd", type=float, default=DEFAULT_KD, help="PD derivative gain.")
    parser.add_argument("--max-force", type=float, default=DEFAULT_MAX_FORCE, help="Max PD force magnitude.")
    parser.add_argument(
        "--dock-distance-threshold",
        type=float,
        default=DEFAULT_DOCKING_DISTANCE_THRESHOLD,
        help="Docking distance threshold (m), used in docking mode.",
    )
    parser.add_argument(
        "--dock-speed-threshold",
        type=float,
        default=DEFAULT_DOCKING_SPEED_THRESHOLD,
        help="Docking speed threshold (m/s), used in docking mode.",
    )
    parser.add_argument(
        "--sensor-noise-std",
        type=float,
        default=DEFAULT_SENSOR_NOISE_STD,
        help="Std dev of Gaussian noise added to measured refuel alignment error (m).",
    )
    parser.add_argument(
        "--alignment-threshold",
        type=float,
        default=DEFAULT_ALIGNMENT_THRESHOLD,
        help="Alignment error threshold (m) used in refuel mode.",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=DEFAULT_REFUEL_SPEED_THRESHOLD,
        help="Speed threshold (m/s) used in refuel mode.",
    )
    parser.add_argument("--csv-output", default=None, help="CSV output path (optional, mode-specific default if omitted).")
    parser.add_argument("--hist-output", default=None, help="Histogram image path (optional, mode-specific default if omitted).")
    parser.add_argument("--scatter-output", default=None, help="Scatter image path (optional, mode-specific default if omitted).")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful for quick troubleshooting runs).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip MuJoCo import and one-shot preflight checks.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=DEFAULT_PREFLIGHT_TIMEOUT_S,
        help="Timeout in seconds for the one-shot preflight subprocess.",
    )
    parser.add_argument(
        "--import-timeout",
        type=float,
        default=DEFAULT_IMPORT_TIMEOUT_S,
        help="Timeout in seconds for the import-only MuJoCo check subprocess.",
    )
    parser.add_argument(
        "--internal-preflight-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-import-check-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbose-preflight",
        action="store_true",
        help="Print stdout/stderr from preflight child processes.",
    )
    return parser.parse_args()


def run_docking_monte_carlo(args):
    rng = np.random.default_rng(args.seed)
    results = []
    initial_positions = []

    for trial_id in range(1, args.trials + 1):
        initial_position = sample_initial_position(rng, mode="docking")
        result = run_one_sim(initial_position, trial_id=trial_id)
        results.append(result)
        initial_positions.append(initial_position)

        print(
            f"trial={trial_id:03d} "
            f"mode=docking "
            f"success={result['success']} "
            f"init_dist={result['initial_distance']:.3f} "
            f"final_dist={result['final_distance']:.3f} "
            f"final_speed={result['final_speed']:.3f}"
        )

    success_flags = np.array([int(r["success"]) for r in results], dtype=int)
    initial_metric = np.array([r["initial_distance"] for r in results], dtype=float)
    final_distances = np.array([r["final_distance"] for r in results], dtype=float)
    final_speeds = np.array([r["final_speed"] for r in results], dtype=float)
    min_distances = np.array([r["min_distance"] for r in results], dtype=float)
    success_times = np.array([r["time_to_dock"] for r in results if r["time_to_dock"] is not None], dtype=float)
    success_rate = 100.0 * success_flags.mean()

    print("\nMonte Carlo summary (docking)")
    print(f"- Trials: {args.trials}")
    print(f"- Successes: {success_flags.sum()} / {args.trials} ({success_rate:.1f}%)")
    print(f"- Mean initial distance: {initial_metric.mean():.4f} m")
    print(f"- Mean final distance: {final_distances.mean():.4f} m")
    print(f"- Mean final speed: {final_speeds.mean():.4f} m/s")
    print(f"- Mean minimum distance: {min_distances.mean():.4f} m")
    if success_times.size > 0:
        print(f"- Mean time to dock (success only): {success_times.mean():.4f} s")
        print(f"- Median time to dock (success only): {np.median(success_times):.4f} s")
    else:
        print("- Time to dock stats: N/A (no successful trials)")

    csv_output, hist_output, scatter_output = resolve_output_paths(args)
    save_results_csv(results, initial_positions, csv_output, mode="docking")
    if not args.no_plots:
        plot_time_histogram(
            success_times,
            hist_output,
            x_label="Time to Dock (s)",
            title="Histogram of Time-to-Dock (Successful Trials)",
        )
        plot_initial_metric_vs_success(
            initial_metric,
            success_flags,
            scatter_output,
            x_label="Initial Distance (m)",
            title="Initial Distance vs Docking Success",
        )

    print("\nSaved outputs")
    print(f"- CSV: {csv_output}")
    if args.no_plots:
        print("- Plots: skipped (--no-plots)")
    else:
        print(f"- Histogram: {hist_output}")
        print(f"- Scatter: {scatter_output}")


def run_refuel_monte_carlo(args):
    rng = np.random.default_rng(args.seed)
    results = []
    initial_positions = []

    for trial_id in range(1, args.trials + 1):
        initial_position = sample_initial_position(rng, mode="refuel")
        result = run_one_refuel_sim(
            initial_position,
            sensor_noise_std=args.sensor_noise_std,
            trial_id=trial_id,
        )
        results.append(result)
        initial_positions.append(initial_position)

        print(
            f"trial={trial_id:03d} "
            f"mode=refuel "
            f"success={result['success']} "
            f"init_err={result['initial_alignment_error']:.3f} "
            f"final_err={result['final_alignment_error']:.3f} "
            f"final_speed={result['final_speed']:.3f}"
        )

    success_flags = np.array([int(r["success"]) for r in results], dtype=int)
    initial_metric = np.array([r["initial_alignment_error"] for r in results], dtype=float)
    final_errors = np.array([r["final_alignment_error"] for r in results], dtype=float)
    final_speeds = np.array([r["final_speed"] for r in results], dtype=float)
    min_errors = np.array([r["min_alignment_error"] for r in results], dtype=float)
    success_times = np.array([r["time_to_align"] for r in results if r["time_to_align"] is not None], dtype=float)
    success_rate = 100.0 * success_flags.mean()

    print("\nMonte Carlo summary (refuel)")
    print(f"- Trials: {args.trials}")
    print(f"- Successes: {success_flags.sum()} / {args.trials} ({success_rate:.1f}%)")
    print(f"- Mean initial alignment error: {initial_metric.mean():.4f} m")
    print(f"- Mean final alignment error: {final_errors.mean():.4f} m")
    print(f"- Mean final speed: {final_speeds.mean():.4f} m/s")
    print(f"- Mean minimum alignment error: {min_errors.mean():.4f} m")
    if success_times.size > 0:
        print(f"- Mean time to align (success only): {success_times.mean():.4f} s")
        print(f"- Median time to align (success only): {np.median(success_times):.4f} s")
    else:
        print("- Time to align stats: N/A (no successful trials)")

    csv_output, hist_output, scatter_output = resolve_output_paths(args)
    save_results_csv(results, initial_positions, csv_output, mode="refuel")
    if not args.no_plots:
        plot_time_histogram(
            success_times,
            hist_output,
            x_label="Time to Align (s)",
            title="Histogram of Time-to-Align (Successful Trials)",
        )
        plot_initial_metric_vs_success(
            initial_metric,
            success_flags,
            scatter_output,
            x_label="Initial Alignment Error (m)",
            title="Initial Alignment Error vs Refuel Success",
        )

    print("\nSaved outputs")
    print(f"- CSV: {csv_output}")
    if args.no_plots:
        print("- Plots: skipped (--no-plots)")
    else:
        print(f"- Histogram: {hist_output}")
        print(f"- Scatter: {scatter_output}")


def main():
    args = parse_args()
    configure_runtime(args)

    if args.internal_preflight_child:
        run_preflight_child_once(args)
        return
    if args.internal_import_check_child:
        run_import_check_child_once()
        return

    if args.trials <= 0:
        raise ValueError("--trials must be a positive integer")
    if SIM_DURATION_S <= 0:
        raise ValueError("--sim-duration must be positive")
    if args.import_timeout <= 0:
        raise ValueError("--import-timeout must be positive")
    if args.preflight_timeout <= 0:
        raise ValueError("--preflight-timeout must be positive")
    if args.sensor_noise_std < 0:
        raise ValueError("--sensor-noise-std must be non-negative")
    if args.alignment_threshold <= 0:
        raise ValueError("--alignment-threshold must be positive")
    if args.speed_threshold <= 0:
        raise ValueError("--speed-threshold must be positive")

    if not args.skip_preflight:
        try:
            run_import_check(args)
            run_preflight_smoke_test(args)
        except RuntimeError as exc:
            raise SystemExit(f"Preflight failed: {exc}") from exc

    if args.mode == "docking":
        run_docking_monte_carlo(args)
    else:
        run_refuel_monte_carlo(args)


if __name__ == "__main__":
    main()
