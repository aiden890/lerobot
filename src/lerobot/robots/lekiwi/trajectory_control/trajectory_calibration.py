# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trajectory calibration module for analyzing and validating joint angle ranges."""

# Default safe joint limits (adjust based on your robot's actual limits)
DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "arm_shoulder_pan.pos": (-90.0, 90.0),
    "arm_shoulder_lift.pos": (-90.0, 90.0),
    "arm_elbow_flex.pos": (-90.0, 90.0),
    "arm_wrist_flex.pos": (-90.0, 90.0),
    "arm_wrist_roll.pos": (-180.0, 180.0),
    "arm_gripper.pos": (0.0, 100.0),
}


def check_trajectory_range(trajectory: list[dict]) -> dict[str, tuple[float, float]]:
    """
    Analyze the joint angle ranges in a trajectory.

    Args:
        trajectory: List of action dictionaries

    Returns:
        Dictionary mapping joint names to (min_value, max_value) tuples
    """
    if not trajectory:
        return {}

    ranges: dict[str, tuple[float, float]] = {}
    joint_keys = [k for k in trajectory[0].keys() if k.endswith(".pos")]

    for key in joint_keys:
        values = [action[key] for action in trajectory if key in action]
        if values:
            ranges[key] = (min(values), max(values))

    return ranges


def print_trajectory_info(trajectory: list[dict], fps: float = 30.0) -> None:
    """
    Print detailed information about a trajectory.

    Args:
        trajectory: List of action dictionaries
        fps: Frames per second for duration calculation
    """
    if not trajectory:
        print("Empty trajectory!")
        return

    ranges = check_trajectory_range(trajectory)
    duration = len(trajectory) / fps

    print("=== Trajectory Info ===")
    print(f"Total frames: {len(trajectory)}")
    print(f"Expected duration: {duration:.2f}s (@ {fps} fps)")
    print(f"\n=== Joint Ranges ===")

    for joint, (min_val, max_val) in sorted(ranges.items()):
        range_size = max_val - min_val
        print(f"  {joint}: {min_val:.2f} ~ {max_val:.2f} (range: {range_size:.2f})")

    # Print initial and final states
    print(f"\n=== Initial State ===")
    for key in sorted(trajectory[0].keys()):
        if key.endswith(".pos"):
            print(f"  {key}: {trajectory[0][key]:.2f}")

    print(f"\n=== Final State ===")
    for key in sorted(trajectory[-1].keys()):
        if key.endswith(".pos"):
            print(f"  {key}: {trajectory[-1][key]:.2f}")


def validate_trajectory(
    trajectory: list[dict],
    limits: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """
    Validate that trajectory stays within safe joint limits.

    Args:
        trajectory: List of action dictionaries
        limits: Joint limits dictionary. Uses DEFAULT_JOINT_LIMITS if None.

    Returns:
        List of warning messages (empty if all joints are within limits)
    """
    if limits is None:
        limits = DEFAULT_JOINT_LIMITS

    warnings: list[str] = []
    ranges = check_trajectory_range(trajectory)

    for joint, (min_val, max_val) in ranges.items():
        if joint in limits:
            limit_min, limit_max = limits[joint]
            if min_val < limit_min:
                warnings.append(f"WARNING: {joint}: min value {min_val:.2f} below limit {limit_min:.2f}")
            if max_val > limit_max:
                warnings.append(f"WARNING: {joint}: max value {max_val:.2f} above limit {limit_max:.2f}")

    return warnings


def get_trajectory_summary(trajectory: list[dict], fps: float = 30.0) -> dict:
    """
    Get a summary dictionary of trajectory information.

    Args:
        trajectory: List of action dictionaries
        fps: Frames per second

    Returns:
        Summary dictionary with frames, duration, ranges, etc.
    """
    ranges = check_trajectory_range(trajectory)
    warnings = validate_trajectory(trajectory)

    return {
        "frames": len(trajectory),
        "duration_sec": len(trajectory) / fps,
        "fps": fps,
        "joint_ranges": ranges,
        "initial_state": trajectory[0].copy() if trajectory else {},
        "final_state": trajectory[-1].copy() if trajectory else {},
        "warnings": warnings,
        "is_valid": len(warnings) == 0,
    }


def compare_trajectories(traj1: list[dict], traj2: list[dict], fps: float = 30.0) -> dict:
    """
    Compare two trajectories.

    Args:
        traj1: First trajectory
        traj2: Second trajectory
        fps: Frames per second

    Returns:
        Comparison dictionary
    """
    ranges1 = check_trajectory_range(traj1)
    ranges2 = check_trajectory_range(traj2)

    all_joints = set(ranges1.keys()) | set(ranges2.keys())

    comparison = {
        "traj1_frames": len(traj1),
        "traj2_frames": len(traj2),
        "traj1_duration": len(traj1) / fps,
        "traj2_duration": len(traj2) / fps,
        "joint_comparison": {},
    }

    for joint in sorted(all_joints):
        joint_info = {}
        if joint in ranges1:
            joint_info["traj1"] = ranges1[joint]
        if joint in ranges2:
            joint_info["traj2"] = ranges2[joint]
        comparison["joint_comparison"][joint] = joint_info

    return comparison


def clamp_trajectory_to_limits(
    trajectory: list[dict],
    limits: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    """
    Clamp trajectory values to stay within joint limits.

    Args:
        trajectory: List of action dictionaries
        limits: Joint limits. Uses DEFAULT_JOINT_LIMITS if None.

    Returns:
        New trajectory with clamped values
    """
    if limits is None:
        limits = DEFAULT_JOINT_LIMITS

    clamped_trajectory = []
    for action in trajectory:
        clamped_action = action.copy()
        for joint, (limit_min, limit_max) in limits.items():
            if joint in clamped_action:
                clamped_action[joint] = max(limit_min, min(limit_max, clamped_action[joint]))
        clamped_trajectory.append(clamped_action)

    return clamped_trajectory
