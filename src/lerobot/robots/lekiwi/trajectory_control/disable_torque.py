#!/usr/bin/env python
"""
Disable arm torque for manual positioning.

Usage:
    python -m lerobot.robots.lekiwi.trajectory_control.disable_torque
    python -m lerobot.robots.lekiwi.trajectory_control.disable_torque --ip 192.168.20.199
"""

import argparse

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

from .config import get_config


def main():
    parser = argparse.ArgumentParser(description="Disable arm torque")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP address")
    args = parser.parse_args()

    config = get_config()
    ip = args.ip or config.robot.remote_ip

    print(f"Connecting to robot at {ip}...")
    robot_config = LeKiwiClientConfig(remote_ip=ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to robot!")
        return

    print("Robot connected.")
    print("\nTo disable torque, you need to run this on the robot host (Raspberry Pi).")
    print("The LeKiwiClient doesn't have direct motor control - it sends commands to the host.")
    print("\nOn the robot host, you can use:")
    print("  from lerobot.robots.lekiwi.lekiwi import LeKiwi")
    print("  robot = LeKiwi(config)")
    print("  robot.bus.disable_torque(robot.arm_motors)")

    robot.disconnect()


if __name__ == "__main__":
    main()
