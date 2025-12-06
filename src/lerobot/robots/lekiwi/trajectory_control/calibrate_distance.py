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
Distance calibration script for trajectory control.

Replays trajectory step-by-step while showing camera feed.
Press keys to record near/far points with their timestep and save screenshots.
After recording, manually update config.yaml with the timestep and pixel coordinates.

Usage:
    python -m lerobot.robots.lekiwi.trajectory_control.calibrate_distance
    python -m lerobot.robots.lekiwi.trajectory_control.calibrate_distance --action pick
    python -m lerobot.robots.lekiwi.trajectory_control.calibrate_distance --action place --ip 192.168.20.199

Controls:
    SPACE: Pause/Resume playback
    LEFT/RIGHT: Step backward/forward (when paused)
    'n': Record current timestep as NEAR point + save screenshot
    'f': Record current timestep as FAR point + save screenshot
    's': Save screenshot
    'r': Restart trajectory from beginning
    'q': Quit
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

from .camera_calibration import draw_calibration_overlay
from .color_detection import convert_to_bgr
from .config import get_config
from .trajectory_utils import load_trajectory_from_dataset


def main():
    parser = argparse.ArgumentParser(description="Distance calibration for trajectory control")
    parser.add_argument("--action", type=str, choices=["pick", "place"], default="pick",
                        help="Action to calibrate (pick or place)")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP address")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback FPS (slower for calibration)")
    args = parser.parse_args()

    config = get_config()
    ip = args.ip or config.robot.remote_ip

    # Load trajectory
    dataset_name = config.datasets.approach_pick if args.action == "pick" else config.datasets.approach_place
    print(f"Loading trajectory from: {dataset_name}")

    try:
        trajectory = load_trajectory_from_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load trajectory: {e}")
        return

    print(f"Trajectory loaded: {len(trajectory)} frames")

    # Connect to robot
    print(f"\nConnecting to robot at {ip}...")
    robot_config = LeKiwiClientConfig(remote_ip=ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to robot!")
        return

    print("Robot connected.")

    # Create output directory for screenshots
    output_dir = Path("calibration_screenshots")
    output_dir.mkdir(exist_ok=True)

    # Calibration state
    current_frame = 0
    paused = True  # Start paused
    near_point = {"timestep": 0, "screenshot": None}
    far_point = {"timestep": 0, "screenshot": None}

    window_name = "Distance Calibration"
    cv2.namedWindow(window_name)

    print("\n" + "=" * 60)
    print(f"Distance Calibration - {args.action.upper()}")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE: Pause/Resume playback")
    print("  LEFT/RIGHT: Step backward/forward (when paused)")
    print("  'n': Record current timestep as NEAR point")
    print("  'f': Record current timestep as FAR point")
    print("  's': Save screenshot")
    print("  'r': Restart from beginning")
    print("  'q': Quit")
    print("=" * 60)
    print("\nStarting in PAUSED mode. Press SPACE to start playback.")
    print(f"Frame: 0/{len(trajectory) - 1}")

    try:
        while True:
            # Get camera image
            obs = robot.get_observation()
            image = obs.get(config.camera.obs_key)

            if image is None:
                print(f"No camera image! (key: {config.camera.obs_key})")
                break

            # Convert to BGR for OpenCV
            image = convert_to_bgr(image, config.camera.color_format)
            display = draw_calibration_overlay(image)
            h, w = display.shape[:2]

            # Draw current frame info
            status = "PAUSED" if paused else "PLAYING"
            cv2.putText(display, f"Frame: {current_frame}/{len(trajectory) - 1} [{status}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, f"Action: {args.action.upper()}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw recorded points
            near_text = f"NEAR: frame {near_point['timestep']}" if near_point["timestep"] > 0 else "NEAR: not set (press 'n')"
            far_text = f"FAR: frame {far_point['timestep']}" if far_point["timestep"] > 0 else "FAR: not set (press 'f')"

            near_color = (0, 255, 0) if near_point["timestep"] > 0 else (128, 128, 128)
            far_color = (0, 255, 0) if far_point["timestep"] > 0 else (128, 128, 128)

            cv2.putText(display, near_text, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, near_color, 2)
            cv2.putText(display, far_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, far_color, 2)

            cv2.imshow(window_name, display)

            # Handle key input
            key = cv2.waitKey(1 if not paused else 50) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):  # SPACE - toggle pause
                paused = not paused
                print(f"\n{'PAUSED' if paused else 'PLAYING'} at frame {current_frame}")
            elif key == 81 or key == 2:  # LEFT arrow
                if paused and current_frame > 0:
                    current_frame -= 1
                    robot.send_action(trajectory[current_frame])
                    print(f"\rFrame: {current_frame}/{len(trajectory) - 1}", end="")
            elif key == 83 or key == 3:  # RIGHT arrow
                if paused and current_frame < len(trajectory) - 1:
                    current_frame += 1
                    robot.send_action(trajectory[current_frame])
                    print(f"\rFrame: {current_frame}/{len(trajectory) - 1}", end="")
            elif key == ord("n"):  # Record NEAR point
                near_point["timestep"] = current_frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"{args.action}_near_{timestamp}.png"
                cv2.imwrite(str(filename), display)
                near_point["screenshot"] = str(filename)
                print(f"\n>>> NEAR point recorded: frame {current_frame}")
                print(f"    Screenshot saved: {filename}")
            elif key == ord("f"):  # Record FAR point
                far_point["timestep"] = current_frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"{args.action}_far_{timestamp}.png"
                cv2.imwrite(str(filename), display)
                far_point["screenshot"] = str(filename)
                print(f"\n>>> FAR point recorded: frame {current_frame}")
                print(f"    Screenshot saved: {filename}")
            elif key == ord("s"):  # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"{args.action}_frame{current_frame}_{timestamp}.png"
                cv2.imwrite(str(filename), display)
                print(f"\nScreenshot saved: {filename}")
            elif key == ord("r"):  # Restart
                current_frame = 0
                paused = True
                robot.send_action(trajectory[0])
                print("\nRestarted from beginning (PAUSED)")

            # Playback logic
            if not paused:
                t0 = time.perf_counter()
                robot.send_action(trajectory[current_frame])

                if current_frame < len(trajectory) - 1:
                    current_frame += 1
                else:
                    paused = True
                    print(f"\nReached end of trajectory. PAUSED at frame {current_frame}")

                # Control playback speed
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, 1.0 / args.fps - elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 60)
    print("Recording Complete")
    print("=" * 60)
    print(f"\nRecorded points for {args.action.upper()}:")
    print(f"  NEAR: frame {near_point['timestep']}")
    if near_point.get("screenshot"):
        print(f"        screenshot: {near_point['screenshot']}")
    print(f"  FAR:  frame {far_point['timestep']}")
    if far_point.get("screenshot"):
        print(f"        screenshot: {far_point['screenshot']}")

    print("\n" + "-" * 60)
    print("Next steps:")
    print("1. Open the screenshots and find the gripper tip pixel coordinates")
    print("2. Update config.yaml with the timestep and pixel_distance values:")
    print(f"   distance.{args.action}.near.timestep = {near_point['timestep']}")
    print(f"   distance.{args.action}.far.timestep = {far_point['timestep']}")
    print("-" * 60)

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
