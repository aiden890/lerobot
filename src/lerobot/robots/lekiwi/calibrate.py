#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Calibrate SO100 Follower arm motors.

Usage:
    python -m lerobot.robots.lekiwi.calibrate --port=/dev/tty.usbmodem5AAF2199371 --id=so100_follower
"""

import argparse
import logging

from lerobot.robots.so100_follower import SO100Follower
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig


def main():
    parser = argparse.ArgumentParser(description="Calibrate SO100 Follower arm")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem5AAF2199371",
        help="Serial port for the motor bus (default: /dev/tty.usbmodem5AAF2199371)",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="so100_follower",
        help="Robot ID for saving calibration file (default: so100_follower)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Starting SO100 Follower calibration on port: {args.port}")
    print(f"Robot ID: {args.id}")

    # Create config and robot
    config = SO100FollowerConfig(port=args.port, id=args.id)
    robot = SO100Follower(config)

    # Connect will automatically ask whether to calibrate
    # If calibration file exists, it will ask whether to use it or run new calibration
    robot.connect(calibrate=True)

    print("\nCalibration completed!")
    print(f"Calibration file saved for robot ID: {args.id}")

    # Disconnect
    robot.disconnect()


if __name__ == "__main__":
    main()
