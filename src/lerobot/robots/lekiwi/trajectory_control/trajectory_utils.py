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

"""Trajectory loading, playback, and interpolation utilities."""

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait


def load_trajectory_from_dataset(dataset_name: str, episode_idx: int = 0) -> list[dict]:
    """
    Load action trajectory from LeRobotDataset.

    Uses the same approach as replay.py for consistency.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "user/trajectory_dataset")
        episode_idx: Episode index to load (default: 0)

    Returns:
        List of action dictionaries, each containing joint positions
    """
    dataset = LeRobotDataset(dataset_name, episodes=[episode_idx])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode_idx)
    actions = episode_frames.select_columns(ACTION)

    trajectory = []
    for idx in range(len(episode_frames)):
        action = {name: float(actions[idx][ACTION][i]) for i, name in enumerate(dataset.features[ACTION]["names"])}
        trajectory.append(action)

    return trajectory


def move_shoulder_pan(
    robot,
    target_angle: float,
    speed_deg_per_sec: float = 30.0,
    fps: float = 30.0,
) -> int:
    """
    Move shoulder pan to target angle at constant speed, keeping other joints fixed.

    Args:
        robot: Robot instance with get_observation and send_action methods
        target_angle: Target shoulder pan angle in degrees
        speed_deg_per_sec: Movement speed in degrees per second
        fps: Frames per second

    Returns:
        Number of frames played
    """
    current = robot.get_observation()
    current_angle = current.get("arm_shoulder_pan.pos", 0.0)
    angle_diff = abs(target_angle - current_angle)

    if angle_diff < 0.5:  # Already at target
        return 0

    # Calculate number of steps based on constant speed
    duration = angle_diff / speed_deg_per_sec
    steps = max(1, int(duration * fps))

    for i in range(steps):
        t0 = time.perf_counter()
        alpha = (i + 1) / steps

        # Build action from current position, only changing shoulder pan
        action = {k: v for k, v in current.items() if k.endswith(".pos")}
        action["arm_shoulder_pan.pos"] = current_angle * (1 - alpha) + target_angle * alpha

        # Keep base velocity at zero
        for k in ["x.vel", "y.vel", "theta.vel"]:
            action[k] = 0.0

        robot.send_action(action)
        busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    return steps


def play_trajectory_until(
    robot,
    trajectory: list[dict],
    ratio: float,
    fps: float = 30.0,
    shoulder_pan_angle: float | None = None,
    shoulder_pan_speed: float = 30.0,
) -> int:
    """
    Play trajectory up to a ratio (0.0 to 1.0) of its length.

    If shoulder_pan_angle is provided, first moves shoulder pan to target angle
    at constant speed, then plays the arm trajectory with shoulder pan fixed.

    Args:
        robot: Robot instance with send_action method
        trajectory: List of action dictionaries
        ratio: Ratio of trajectory to play (0.0 = start only, 1.0 = full trajectory)
        fps: Frames per second for playback
        shoulder_pan_angle: If provided, move shoulder pan to this angle FIRST, then play trajectory
        shoulder_pan_speed: Speed for shoulder pan movement in degrees per second

    Returns:
        Number of frames played (including shoulder pan movement)
    """
    ratio = max(0.0, min(1.0, ratio))
    target_idx = int(ratio * (len(trajectory) - 1))
    total_frames = 0

    # Phase 1: Move shoulder pan to target angle first (if specified)
    if shoulder_pan_angle is not None:
        pan_frames = move_shoulder_pan(robot, shoulder_pan_angle, shoulder_pan_speed, fps)
        total_frames += pan_frames

    # Phase 2: Play trajectory with shoulder pan fixed at target angle
    for i in range(target_idx + 1):
        t0 = time.perf_counter()
        action = trajectory[i].copy()

        # Keep shoulder pan fixed at target angle during trajectory playback
        if shoulder_pan_angle is not None and "arm_shoulder_pan.pos" in action:
            action["arm_shoulder_pan.pos"] = shoulder_pan_angle

        robot.send_action(action)
        busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    total_frames += target_idx + 1
    return total_frames


def play_full_trajectory(
    robot,
    trajectory: list[dict],
    fps: float = 30.0,
    shoulder_pan_angle: float | None = None,
) -> int:
    """
    Play the entire trajectory.

    Args:
        robot: Robot instance with send_action method
        trajectory: List of action dictionaries
        fps: Frames per second for playback
        shoulder_pan_angle: If provided, override arm_shoulder_pan.pos with this angle (degrees)

    Returns:
        Number of frames played
    """
    return play_trajectory_until(robot, trajectory, ratio=1.0, fps=fps, shoulder_pan_angle=shoulder_pan_angle)


def move_to_initial(robot, initial_state: dict, steps: int = 50, fps: float = 30.0) -> None:
    """
    Smoothly return to initial state using linear interpolation.

    Args:
        robot: Robot instance with get_observation and send_action methods
        initial_state: Target initial state dictionary (from get_observation)
        steps: Number of interpolation steps
        fps: Frames per second
    """
    current = robot.get_observation()

    # Get position keys from initial_state
    pos_keys = [k for k in initial_state if k.endswith(".pos")]

    for i in range(steps):
        t0 = time.perf_counter()
        alpha = (i + 1) / steps  # 0 -> 1 increasing

        interpolated = {}
        for key in pos_keys:
            # Linear interpolation for position values
            current_val = current.get(key, initial_state[key])
            target_val = initial_state[key]
            interpolated[key] = current_val * (1 - alpha) + target_val * alpha

        # Keep base velocity at zero
        for k in ["x.vel", "y.vel", "theta.vel"]:
            interpolated[k] = 0.0

        robot.send_action(interpolated)
        busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    # Send exact target position as final action to ensure accuracy
    final_action = {k: initial_state[k] for k in pos_keys}
    for k in ["x.vel", "y.vel", "theta.vel"]:
        final_action[k] = 0.0
    robot.send_action(final_action)


def interpolate_action(start: dict, end: dict, alpha: float) -> dict:
    """
    Linearly interpolate between two action states.

    Args:
        start: Starting action dictionary
        end: Ending action dictionary
        alpha: Interpolation factor (0.0 = start, 1.0 = end)

    Returns:
        Interpolated action dictionary
    """
    alpha = max(0.0, min(1.0, alpha))
    action = {}

    for key in end:
        if key.endswith(".pos"):
            start_val = start.get(key, end[key])
            action[key] = start_val * (1 - alpha) + end[key] * alpha
        elif key.endswith(".vel"):
            action[key] = 0.0

    return action


def get_trajectory_duration(trajectory: list[dict], fps: float = 30.0) -> float:
    """
    Calculate trajectory duration in seconds.

    Args:
        trajectory: List of action dictionaries
        fps: Frames per second

    Returns:
        Duration in seconds
    """
    return len(trajectory) / fps


def get_trajectory_frame_at_ratio(trajectory: list[dict], ratio: float) -> dict:
    """
    Get a specific frame from trajectory at the given ratio.

    Args:
        trajectory: List of action dictionaries
        ratio: Position in trajectory (0.0 to 1.0)

    Returns:
        Action dictionary at the specified ratio
    """
    ratio = max(0.0, min(1.0, ratio))
    idx = int(ratio * (len(trajectory) - 1))
    return trajectory[idx].copy()


def close_gripper(robot, target: float = 0.0, duration: float = 1.0, fps: float = 30.0) -> None:
    """
    Slowly close the gripper to target value.

    Args:
        robot: Robot instance
        target: Target gripper position (0.0 = fully closed)
        duration: Duration in seconds
        fps: Frames per second
    """
    current = robot.get_observation()
    start_gripper = current.get("arm_gripper.pos", 100.0)
    steps = int(duration * fps)

    for i in range(steps):
        t0 = time.perf_counter()
        alpha = (i + 1) / steps

        # Build action from current position, only changing gripper
        action = {k: v for k, v in current.items() if k.endswith(".pos")}
        action["arm_gripper.pos"] = start_gripper * (1 - alpha) + target * alpha

        # Keep base velocity at zero
        for k in ["x.vel", "y.vel", "theta.vel"]:
            action[k] = 0.0

        robot.send_action(action)
        busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))


def open_gripper(robot, target: float = 100.0, duration: float = 1.0, fps: float = 30.0) -> None:
    """
    Slowly open the gripper to target value.

    Args:
        robot: Robot instance
        target: Target gripper position (100.0 = fully open)
        duration: Duration in seconds
        fps: Frames per second
    """
    current = robot.get_observation()
    start_gripper = current.get("arm_gripper.pos", 0.0)
    steps = int(duration * fps)

    for i in range(steps):
        t0 = time.perf_counter()
        alpha = (i + 1) / steps

        # Build action from current position, only changing gripper
        action = {k: v for k, v in current.items() if k.endswith(".pos")}
        action["arm_gripper.pos"] = start_gripper * (1 - alpha) + target * alpha

        # Keep base velocity at zero
        for k in ["x.vel", "y.vel", "theta.vel"]:
            action[k] = 0.0

        robot.send_action(action)
        busy_wait(max(1.0 / fps - (time.perf_counter() - t0), 0.0))
