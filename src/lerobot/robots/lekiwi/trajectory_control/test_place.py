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
Place action standalone test script.

Detects black region and executes place operation:
1. Detect black region -> calculate distance
2. Play approach trajectory proportionally
3. Open gripper
4. Return to initial state

Usage:
    python test_place.py
    python test_place.py --ip 192.168.20.199
"""

import argparse
import time

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

from .camera_calibration import pixel_to_robot
from .color_detection import convert_to_bgr, detect_black
from .config import get_config
from .trajectory_calibration import print_trajectory_info, validate_trajectory
from .trajectory_utils import (
    load_trajectory_from_dataset,
    move_to_initial,
    open_gripper,
    play_trajectory_until,
)


def test_place(remote_ip: str | None = None, dry_run: bool = False):
    """
    Execute place action test.

    Args:
        remote_ip: Robot IP address (uses config default if None)
        dry_run: If True, only detect and show info without executing
    """
    config = get_config()
    ip = remote_ip or config.robot.remote_ip

    print("=" * 50)
    print("Place Action Test")
    print("=" * 50)

    # Load approach trajectory
    print(f"\nLoading approach_place trajectory: {config.datasets.approach_place}")
    approach_trajectory = load_trajectory_from_dataset(config.datasets.approach_place)
    print_trajectory_info(approach_trajectory, config.robot.fps)

    # Validate trajectory
    warnings = validate_trajectory(approach_trajectory)
    if warnings:
        print("\n=== Warnings ===")
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

        # Detect black region
        print("\nDetecting black region...")
        obs = robot.get_observation()
        image = obs.get(config.camera.obs_key)

        if image is None:
            print(f"No camera image available! (key: {config.camera.obs_key})")
            print(f"Available keys: {[k for k in obs.keys() if 'image' in k.lower() or 'front' in k.lower()]}")
            return

        # Convert to BGR for OpenCV processing
        image = convert_to_bgr(image, config.camera.color_format)

        result = detect_black(image)
        if result is None:
            print("Black region not detected!")
            return

        cx, cy, _, _ = result
        h = image.shape[0]
        pixel_distance, shoulder_pan = pixel_to_robot(cx, cy, h)

        # Calculate trajectory ratio from pixel distance using calibration
        dist_cal = config.distance.place
        near_dist = dist_cal.near.pixel_distance
        far_dist = dist_cal.far.pixel_distance
        near_step = dist_cal.near.timestep
        far_step = dist_cal.far.timestep

        # Check if pixel distance is within calibration range
        min_dist = min(near_dist, far_dist)
        max_dist = max(near_dist, far_dist)

        print(f"\nBlack region detected:")
        print(f"  Pixel: ({cx}, {cy})")
        print(f"  Pixel distance: {pixel_distance:.1f}px")
        print(f"  Valid range: {min_dist:.1f}px ~ {max_dist:.1f}px")
        print(f"  Shoulder pan: {shoulder_pan:.2f}")

        if pixel_distance < min_dist or pixel_distance > max_dist:
            print(f"\n[ERROR] Pixel distance {pixel_distance:.1f}px is out of calibration range!")
            if pixel_distance < min_dist:
                print(f"  Target is too CLOSE (min: {min_dist:.1f}px)")
            else:
                print(f"  Target is too FAR (max: {max_dist:.1f}px)")
            print("Action aborted for safety.")
            return

        # Interpolate timestep based on pixel distance
        if far_dist != near_dist:
            t = (pixel_distance - near_dist) / (far_dist - near_dist)
            target_timestep = near_step + t * (far_step - near_step)
        else:
            target_timestep = near_step

        # Convert timestep to ratio
        trajectory_ratio = target_timestep / len(approach_trajectory)
        trajectory_ratio = max(0.0, min(1.0, trajectory_ratio))

        print(f"  Target timestep: {target_timestep:.0f}/{len(approach_trajectory)}")
        print(f"  Trajectory ratio: {trajectory_ratio:.3f}")

        if dry_run:
            print("\n[DRY RUN] Would execute place action with the above parameters.")
            return

        # Confirm execution
        response = input("\nExecute place action? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            return

        # Execute place
        print("\n=== Executing Place ===")

        # 1. Approach (with shoulder pan angle set to target direction)
        print(f"1. Approaching (ratio: {trajectory_ratio:.3f}, shoulder_pan: {shoulder_pan:.1f})...")
        t0 = time.time()
        frames = play_trajectory_until(
            robot, approach_trajectory, trajectory_ratio, config.robot.fps, shoulder_pan_angle=shoulder_pan
        )
        print(f"   Played {frames} frames in {time.time() - t0:.2f}s")

        # 2. Open gripper (release object)
        print("2. Opening gripper...")
        t0 = time.time()
        open_gripper(robot, target=100.0, duration=config.motion.gripper_duration, fps=config.robot.fps)
        print(f"   Completed in {time.time() - t0:.2f}s")

        # 3. Return to initial
        print("3. Returning to initial state...")
        t0 = time.time()
        move_to_initial(robot, initial_state, config.motion.return_steps, config.robot.fps)
        print(f"   Completed in {time.time() - t0:.2f}s")

        print("\n=== Place Complete ===")

    except KeyboardInterrupt:
        print("\nInterrupted by user!")
    finally:
        robot.disconnect()
        print("Disconnected from robot.")


def main():
    parser = argparse.ArgumentParser(description="Place action standalone test")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP address")
    parser.add_argument("--dry-run", action="store_true", help="Only detect, don't execute")
    args = parser.parse_args()
    test_place(args.ip, args.dry_run)


if __name__ == "__main__":
    main()
