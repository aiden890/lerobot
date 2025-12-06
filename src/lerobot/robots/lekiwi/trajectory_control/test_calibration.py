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
Camera calibration visualization and interactive test script.

This script displays the camera feed with calibration reference points
and allows interactive clicking to measure pixel coordinates and shoulder_pan values.

Usage:
    python test_calibration.py
    python test_calibration.py --ip 192.168.20.199

Controls:
    'q': Quit
    's': Save screenshot
    'p': Pink detection mode
    'b': Black detection mode
    'c': Calibration overlay mode
    'm': Mouse click mode (click to measure)
    Left click: Show pixel distance and shoulder_pan at clicked point
"""

import argparse

import cv2

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

from .camera_calibration import (
    draw_calibration_overlay,
    pixel_to_robot,
    visualize_detection,
)
from .color_detection import (
    TargetColor,
    convert_to_bgr,
    detect_pink,
    visualize_color_detection,
)
from .config import get_config


# Global state for mouse callback
_last_click_info = None


def mouse_callback(event, x, y, flags, param):
    """Handle mouse click events."""
    global _last_click_info
    if event == cv2.EVENT_LBUTTONDOWN:
        image_height = param.get("image_height", 480)
        pixel_distance, shoulder_pan = pixel_to_robot(x, y, image_height)
        _last_click_info = {
            "pixel": (x, y),
            "pixel_distance": pixel_distance,
            "shoulder_pan": shoulder_pan,
        }
        print(f"\n>>> Click at ({x}, {y})")
        print(f"    Pixel distance: {pixel_distance:.0f}px")
        print(f"    Shoulder pan: {shoulder_pan:.2f}")


def main(remote_ip: str | None = None):
    """Run camera calibration visualization test."""
    global _last_click_info

    config = get_config()
    ip = remote_ip or config.robot.remote_ip

    print(f"Connecting to robot at {ip}...")
    robot_config = LeKiwiClientConfig(remote_ip=ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to robot!")
        return

    print("\n" + "=" * 60)
    print("Camera Calibration Test")
    print("=" * 60)
    print("\nCurrent calibration settings:")
    print(f"  Robot origin offset: {config.calibration.robot_origin_offset_px}px below image")
    print(f"  Image center X: {config.calibration.image_center_x}")
    print("\nDirection calibration points:")
    for dp in config.calibration.direction_points:
        print(f"  pixel_x={dp.pixel_x} -> shoulder_pan={dp.shoulder_pan:.2f}")
    print("\n" + "=" * 60)
    print("Controls:")
    print("  'q': Quit")
    print("  's': Save screenshot")
    print("  'p': Pink detection mode")
    print("  'b': Black detection mode")
    print("  'c': Calibration overlay mode")
    print("  'm': Mouse click mode (click to measure)")
    print("  Left click: Show pixel distance and shoulder_pan")
    print("=" * 60 + "\n")

    mode = "calibration"  # calibration, pink, black, mouse
    window_name = "Calibration Test"

    # Create window
    cv2.namedWindow(window_name)

    try:
        while True:
            obs = robot.get_observation()
            image = obs.get(config.camera.obs_key)

            if image is None:
                print(f"No camera image available! (key: {config.camera.obs_key})")
                print(f"Available keys: {[k for k in obs.keys() if 'image' in k.lower() or 'front' in k.lower()]}")
                break

            # Convert to BGR for OpenCV processing
            image = convert_to_bgr(image, config.camera.color_format)
            h, w = image.shape[:2]

            # Set mouse callback with image height
            cv2.setMouseCallback(window_name, mouse_callback, {"image_height": h})

            # Select display mode
            if mode == "calibration":
                display = draw_calibration_overlay(image)

                # Try to detect pink and show info
                result = detect_pink(image)
                if result:
                    cx, cy, _, _ = result
                    pixel_distance, shoulder_pan = pixel_to_robot(cx, cy, h)
                    display = visualize_detection(image, cx, cy, pixel_distance, shoulder_pan)
                    print(f"\rPink: ({cx}, {cy}) -> dist={pixel_distance:.0f}px, pan={shoulder_pan:.2f}    ", end="")
                else:
                    print("\rNo pink object detected                                    ", end="")

            elif mode == "pink":
                display = visualize_color_detection(image, TargetColor.PINK)

            elif mode == "black":
                display = visualize_color_detection(image, TargetColor.BLACK)

            elif mode == "mouse":
                display = draw_calibration_overlay(image)

                # Show last click info if available
                if _last_click_info:
                    px, py = _last_click_info["pixel"]
                    dist = _last_click_info["pixel_distance"]
                    pan = _last_click_info["shoulder_pan"]

                    # Draw clicked point
                    cv2.circle(display, (px, py), 10, (0, 255, 0), -1)
                    cv2.circle(display, (px, py), 12, (255, 255, 255), 2)

                    # Draw line from origin to clicked point
                    origin_x = config.calibration.image_center_x
                    cv2.line(display, (origin_x, h), (px, py), (0, 255, 0), 2)

                    # Show info
                    info_text = f"Click: ({px}, {py}) -> dist={dist:.0f}px, pan={pan:.2f}"
                    cv2.putText(display, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Instructions
                cv2.putText(display, "Click anywhere to measure", (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show mode label
            mode_colors = {
                "calibration": (255, 255, 0),
                "pink": (255, 0, 255),
                "black": (128, 128, 128),
                "mouse": (0, 255, 0),
            }
            cv2.putText(display, f"Mode: {mode}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_colors.get(mode, (255, 255, 255)), 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                filename = f"calibration_{mode}.png"
                cv2.imwrite(filename, display)
                print(f"\nScreenshot saved: {filename}")
            elif key == ord("p"):
                mode = "pink"
                print("\nSwitched to pink detection mode")
            elif key == ord("b"):
                mode = "black"
                print("\nSwitched to black detection mode")
            elif key == ord("c"):
                mode = "calibration"
                print("\nSwitched to calibration mode")
            elif key == ord("m"):
                mode = "mouse"
                _last_click_info = None
                print("\nSwitched to mouse click mode - click anywhere to measure")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Disconnected from robot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration visualization test")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP address")
    args = parser.parse_args()
    main(args.ip)
