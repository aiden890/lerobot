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
Navigate LeKiwi to specific positions using keyboard input.

Usage:
    python -m lerobot.robots.lekiwi.goto_position
"""

import logging
import math
import time

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

# Force logging configuration (even if already initialized)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Python 3.8+ : Override existing configuration
)
logger = logging.getLogger(__name__)


class LeKiwiNavigator:
    """Simple position-based navigator for LeKiwi."""

    def __init__(self, robot: LeKiwiClient, control_hz: float = 10.0):
        self.robot = robot
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

        # Current position (initialized as 0, 0, 0)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # orientation in radians

        # Control parameters
        self.linear_speed = 0.2  # m/s for forward motion
        self.angular_speed = 60.0  # deg/s for rotation
        self.position_tolerance = 0.05  # 5cm tolerance
        self.angle_tolerance = 0.1  # ~5.7 degrees tolerance

    def get_current_position(self) -> tuple[float, float, float]:
        """Return current estimated position (x, y, theta)."""
        return self.x, self.y, self.theta

    def reset_position(self) -> None:
        """Reset current position to origin (0, 0, 0)."""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        logger.info("Position reset to (0, 0, 0)")

    def _update_odometry(self, vx: float, vy: float, vtheta: float) -> None:
        """Update estimated position based on commanded velocities."""
        # Simple dead-reckoning odometry
        dx = (vx * math.cos(self.theta) - vy * math.sin(self.theta)) * self.dt
        dy = (vx * math.sin(self.theta) + vy * math.cos(self.theta)) * self.dt
        dtheta = vtheta * self.dt

        self.x += dx
        self.y += dy
        self.theta += dtheta

        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def goto_position(self, target_x: float, target_y: float) -> bool:
        """
        Navigate to target position (x, y) from current position.

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Navigating from ({self.x:.2f}, {self.y:.2f}) to ({target_x:.2f}, {target_y:.2f})")

        # Calculate required movement
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)

        logger.info(f"Distance: {distance:.3f}m, Target angle: {math.degrees(target_angle):.1f}°")

        # Step 1: Rotate to face target
        if not self._rotate_to_angle(target_angle):
            logger.error("Failed to rotate to target angle")
            return False

        # Step 2: Move forward to target
        if not self._move_distance(distance):
            logger.error("Failed to move to target")
            return False

        logger.info(f"Reached target position ({target_x:.2f}, {target_y:.2f})")
        return True

    def _rotate_to_angle(self, target_angle: float) -> bool:
        """Rotate robot to face target angle."""
        angle_error = target_angle - self.theta
        # Normalize to [-pi, pi]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        logger.info(f"Rotating {math.degrees(angle_error):.1f}°")

        while abs(angle_error) > self.angle_tolerance:
            # Determine rotation direction
            rotation_sign = 1 if angle_error > 0 else -1

            # Send rotation command using velocity format
            angular_vel = rotation_sign * self.angular_speed  # deg/s
            action = {
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": angular_vel,
            }

            self.robot.send_action(action)

            # Update odometry (convert deg/s to rad/s)
            vtheta_rad = math.radians(angular_vel)
            self._update_odometry(0, 0, vtheta_rad)

            # Recalculate error
            angle_error = target_angle - self.theta
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

            time.sleep(self.dt)

        # Stop rotation
        self._stop()
        logger.info(f"Rotation complete. Current angle: {math.degrees(self.theta):.1f}°")
        return True

    def _move_distance(self, distance: float) -> bool:
        """Move robot forward by specified distance."""
        logger.info(f"Moving forward {distance:.3f}m")

        moved = 0.0
        while moved < distance - self.position_tolerance:
            # Calculate remaining distance
            remaining = distance - moved

            # Move forward using velocity command
            # x.vel is forward/backward velocity in m/s
            action = {
                "x.vel": self.linear_speed,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }

            self.robot.send_action(action)

            # Update odometry
            step_distance = min(self.linear_speed * self.dt, remaining)
            vx = self.linear_speed
            self._update_odometry(vx, 0, 0)
            moved += step_distance

            logger.debug(f"Moved: {moved:.3f}m / {distance:.3f}m")
            time.sleep(self.dt)

        # Stop movement
        self._stop()
        logger.info(f"Movement complete. Position: ({self.x:.2f}, {self.y:.2f})")
        return True

    def _stop(self) -> None:
        """Stop all base motors."""
        action = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        }
        self.robot.send_action(action)
        time.sleep(0.1)


def main():
    # Robot configuration
    robot_config = LeKiwiClientConfig(remote_ip="192.168.20.199", id="lekiwi")
    robot = LeKiwiClient(robot_config)

    # Connect to robot
    logger.info("Connecting to LeKiwi...")
    robot.connect()
    logger.info("Connected!")

    # Create navigator
    navigator = LeKiwiNavigator(robot, control_hz=30.0)

    try:
        print("\n" + "="*60)
        print("LeKiwi Position Navigator")
        print("="*60)
        print("Current position set to (0, 0)")
        print("Commands:")
        print("  - Enter target position as: x y")
        print("  - Type 'reset' to reset position to (0, 0)")
        print("  - Type 'pos' to show current position")
        print("  - Type 'quit' to exit")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("Enter command or target position (x y): ").strip().lower()

                if user_input == "quit" or user_input == "q":
                    logger.info("Exiting...")
                    break

                if user_input == "reset":
                    navigator.reset_position()
                    continue

                if user_input == "pos" or user_input == "position":
                    x, y, theta = navigator.get_current_position()
                    print(f"Current position: ({x:.2f}, {y:.2f}), angle: {math.degrees(theta):.1f}°")
                    continue

                # Parse target position
                parts = user_input.split()
                if len(parts) != 2:
                    print("Invalid input. Please enter two numbers: x y")
                    continue

                try:
                    target_x = float(parts[0])
                    target_y = float(parts[1])
                except ValueError:
                    print("Invalid numbers. Please enter valid coordinates.")
                    continue

                # Navigate to target
                logger.info(f"Target: ({target_x:.2f}, {target_y:.2f})")
                success = navigator.goto_position(target_x, target_y)

                if success:
                    print(f"✓ Successfully reached ({target_x:.2f}, {target_y:.2f})")
                else:
                    print(f"✗ Failed to reach ({target_x:.2f}, {target_y:.2f})")

            except KeyboardInterrupt:
                logger.info("\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                continue

    finally:
        # Stop robot and disconnect
        logger.info("Stopping robot...")
        navigator._stop()
        robot.disconnect()
        logger.info("Disconnected")


if __name__ == "__main__":
    main()
