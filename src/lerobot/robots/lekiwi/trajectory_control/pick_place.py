#!/usr/bin/env python
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

"""
Integrated Pick and Place script.

Complete pick-and-place cycle:
1. Detect pink object -> pick it up
2. Detect black region -> place the object

Usage:
    python pick_place.py
    python pick_place.py --ip 192.168.20.199
    python pick_place.py --mode pick    # Pick only
    python pick_place.py --mode place   # Place only
"""

import argparse
import time

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

from .camera_calibration import pixel_to_real, real_to_normalized
from .color_detection import convert_to_bgr, detect_black, detect_pink
from .config import get_config
from .trajectory_calibration import print_trajectory_info, validate_trajectory
from .trajectory_utils import (
    close_gripper,
    load_trajectory_from_dataset,
    move_to_initial,
    open_gripper,
    play_trajectory_until,
)


def pick_object(robot, approach_trajectory, initial_state, config) -> bool:
    """
    Execute pick operation.

    Args:
        robot: Robot instance
        approach_trajectory: Approach trajectory data
        initial_state: Initial state to return to
        config: Configuration object

    Returns:
        True if successful, False otherwise
    """
    print("\n--- Pick Operation ---")

    # Detect pink object
    print("Detecting pink object...")
    obs = robot.get_observation()
    image = obs.get(config.camera.obs_key)

    if image is None:
        print(f"ERROR: No camera image available! (key: {config.camera.obs_key})")
        print(f"Available keys: {[k for k in obs.keys() if 'image' in k.lower() or 'front' in k.lower()]}")
        return False

    # Convert to BGR for OpenCV processing
    image = convert_to_bgr(image, config.camera.color_format)

    result = detect_pink(image)
    if result is None:
        print("ERROR: Pink object not detected!")
        return False

    cx, cy, _, _ = result
    distance_cm, angle_deg = pixel_to_real(cx, cy)
    norm_dist, _ = real_to_normalized(distance_cm, angle_deg)

    print(f"Pink object: ({cx}, {cy}) -> {distance_cm:.1f}cm, {angle_deg:.1f}deg")

    # 1. Approach (with shoulder pan pointing to object)
    # First moves shoulder pan to target angle, then plays arm trajectory
    print(f"Approaching (ratio: {norm_dist:.3f}, angle: {angle_deg:.1f}deg)...")
    t0 = time.time()
    frames = play_trajectory_until(
        robot,
        approach_trajectory,
        norm_dist,
        config.robot.fps,
        shoulder_pan_angle=angle_deg,
        shoulder_pan_speed=config.motion.shoulder_pan_speed,
    )
    print(f"  Played {frames} frames in {time.time() - t0:.2f}s")

    # 2. Close gripper (target = 0)
    print("Closing gripper...")
    t0 = time.time()
    close_gripper(robot, target=0.0, duration=config.motion.gripper_duration, fps=config.robot.fps)
    print(f"  Completed in {time.time() - t0:.2f}s")

    # 3. Return to initial
    print("Returning to initial state...")
    t0 = time.time()
    move_to_initial(robot, initial_state, config.motion.return_steps, config.robot.fps)
    print(f"  Completed in {time.time() - t0:.2f}s")

    print("--- Pick Complete ---")
    return True


def place_object(robot, approach_trajectory, initial_state, config) -> bool:
    """
    Execute place operation.

    Args:
        robot: Robot instance
        approach_trajectory: Approach trajectory data
        initial_state: Initial state to return to
        config: Configuration object

    Returns:
        True if successful, False otherwise
    """
    print("\n--- Place Operation ---")

    # Detect black region
    print("Detecting black region...")
    obs = robot.get_observation()
    image = obs.get(config.camera.obs_key)

    if image is None:
        print(f"ERROR: No camera image available! (key: {config.camera.obs_key})")
        print(f"Available keys: {[k for k in obs.keys() if 'image' in k.lower() or 'front' in k.lower()]}")
        return False

    # Convert to BGR for OpenCV processing
    image = convert_to_bgr(image, config.camera.color_format)

    result = detect_black(image)
    if result is None:
        print("ERROR: Black region not detected!")
        return False

    cx, cy, _, _ = result
    distance_cm, angle_deg = pixel_to_real(cx, cy)
    norm_dist, _ = real_to_normalized(distance_cm, angle_deg)

    print(f"Black region: ({cx}, {cy}) -> {distance_cm:.1f}cm, {angle_deg:.1f}deg")

    # 1. Approach (with shoulder pan pointing to target)
    # First moves shoulder pan to target angle, then plays arm trajectory
    print(f"Approaching (ratio: {norm_dist:.3f}, angle: {angle_deg:.1f}deg)...")
    t0 = time.time()
    frames = play_trajectory_until(
        robot,
        approach_trajectory,
        norm_dist,
        config.robot.fps,
        shoulder_pan_angle=angle_deg,
        shoulder_pan_speed=config.motion.shoulder_pan_speed,
    )
    print(f"  Played {frames} frames in {time.time() - t0:.2f}s")

    # 2. Open gripper
    print("Opening gripper...")
    t0 = time.time()
    open_gripper(robot, target=config.motion.gripper_open, duration=config.motion.gripper_duration, fps=config.robot.fps)
    print(f"  Completed in {time.time() - t0:.2f}s")

    # 3. Return to initial
    print("Returning to initial state...")
    t0 = time.time()
    move_to_initial(robot, initial_state, config.motion.return_steps, config.robot.fps)
    print(f"  Completed in {time.time() - t0:.2f}s")

    print("--- Place Complete ---")
    return True


def run_pick_place(remote_ip: str | None = None, mode: str = "both"):
    """
    Run pick and/or place operations.

    Args:
        remote_ip: Robot IP address (uses config default if None)
        mode: "pick", "place", or "both"
    """
    config = get_config()
    ip = remote_ip or config.robot.remote_ip

    print("=" * 60)
    print("Pick and Place Operation")
    print(f"Mode: {mode}")
    print("=" * 60)

    # Load trajectories based on mode
    trajectories = {}

    if mode in ("pick", "both"):
        print(f"\nLoading approach_pick: {config.datasets.approach_pick}")
        trajectories["approach_pick"] = load_trajectory_from_dataset(config.datasets.approach_pick)
        print_trajectory_info(trajectories["approach_pick"], config.robot.fps)

    if mode in ("place", "both"):
        print(f"\nLoading approach_place: {config.datasets.approach_place}")
        trajectories["approach_place"] = load_trajectory_from_dataset(config.datasets.approach_place)
        print_trajectory_info(trajectories["approach_place"], config.robot.fps)

    # Validate all trajectories
    warnings = []
    for name, traj in trajectories.items():
        w = validate_trajectory(traj)
        warnings.extend([f"[{name}] {msg}" for msg in w])

    if warnings:
        print("\n=== Trajectory Warnings ===")
        for w in warnings:
            print(f"  {w}")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Connect to robot
    print(f"\nConnecting to robot at {ip}...")
    robot_config = LeKiwiClientConfig(remote_ip=ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to robot!")
        return

    try:
        # Save initial state
        initial_state = robot.get_observation()
        print("Initial state saved.")

        # Confirm execution
        response = input(f"\nExecute {mode} operation? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return

        print("\n" + "=" * 60)
        print("Starting execution...")
        print("=" * 60)

        total_start = time.time()

        # Execute pick
        if mode in ("pick", "both"):
            success = pick_object(
                robot,
                trajectories["approach_pick"],
                initial_state,
                config,
            )
            if not success and mode == "both":
                print("\nPick failed, aborting place operation.")
                return

        # Wait between operations
        if mode == "both":
            print("\nWaiting 1 second before place operation...")
            time.sleep(1.0)

        # Execute place
        if mode in ("place", "both"):
            success = place_object(
                robot,
                trajectories["approach_place"],
                initial_state,
                config,
            )
            if not success:
                print("\nPlace failed.")
                return

        print("\n" + "=" * 60)
        print(f"Operation complete! Total time: {time.time() - total_start:.2f}s")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        print("Attempting to return to initial state...")
        try:
            move_to_initial(robot, initial_state, config.motion.return_steps, config.robot.fps)
        except Exception as e:
            print(f"Failed to return to initial: {e}")
    finally:
        robot.disconnect()
        print("Disconnected from robot.")


def main():
    parser = argparse.ArgumentParser(description="Pick and Place operation")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP address")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["pick", "place", "both"],
        help="Operation mode: pick, place, or both (default: both)",
    )
    args = parser.parse_args()
    run_pick_place(args.ip, args.mode)


if __name__ == "__main__":
    main()
