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
Color calibration script for HSV range tuning.

Click on pixels to see their HSV values and determine color detection ranges.

Usage:
    python -m lerobot.robots.lekiwi.trajectory_control.calibrate_color
    python -m lerobot.robots.lekiwi.trajectory_control.calibrate_color --ip 192.168.20.199

Controls:
    Left click: Show HSV value at clicked point
    'c': Clear recorded points
    'p': Show current pink detection result
    'b': Show current black detection result
    's': Save screenshot
    'q': Quit
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

from .camera_calibration import pixel_to_robot
from .color_detection import convert_to_bgr, detect_black_detailed, detect_pink_detailed, get_roi, reload_config, TargetColor
from .config import get_config


# Store clicked HSV values
_clicked_points: list[dict] = []


def mouse_callback(event, x, y, flags, param):
    """Handle mouse click to show HSV value and robot coordinates."""
    global _clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        image_bgr = param.get("image_bgr")
        if image_bgr is None:
            return

        h_img, w_img = image_bgr.shape[:2]

        # Get BGR value at clicked point
        b, g, r = image_bgr[y, x]

        # Convert single pixel to HSV
        pixel_bgr = np.uint8([[[b, g, r]]])
        pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = pixel_hsv[0, 0]

        # Get robot coordinates (pixel_distance, shoulder_pan)
        pixel_distance, shoulder_pan = pixel_to_robot(x, y, h_img)

        point_info = {
            "pixel": (x, y),
            "bgr": (int(b), int(g), int(r)),
            "hsv": (int(h), int(s), int(v)),
            "pixel_distance": pixel_distance,
            "shoulder_pan": shoulder_pan,
        }
        _clicked_points.append(point_info)

        print(f"\n>>> Click at ({x}, {y})")
        print(f"    BGR: ({b}, {g}, {r})")
        print(f"    HSV: ({h}, {s}, {v})")
        print(f"    Pixel distance: {pixel_distance:.1f}px")
        print(f"    Shoulder pan: {shoulder_pan:.2f}deg")

        # If we have multiple points, suggest a range
        if len(_clicked_points) >= 2:
            h_vals = [p["hsv"][0] for p in _clicked_points]
            s_vals = [p["hsv"][1] for p in _clicked_points]
            v_vals = [p["hsv"][2] for p in _clicked_points]

            # Add some margin
            margin_h = 10
            margin_s = 30
            margin_v = 30

            lower = [
                max(0, min(h_vals) - margin_h),
                max(0, min(s_vals) - margin_s),
                max(0, min(v_vals) - margin_v),
            ]
            upper = [
                min(180, max(h_vals) + margin_h),
                min(255, max(s_vals) + margin_s),
                min(255, max(v_vals) + margin_v),
            ]

            print(f"\n    Suggested HSV range (from {len(_clicked_points)} points):")
            print(f"    lower: [{lower[0]}, {lower[1]}, {lower[2]}]")
            print(f"    upper: [{upper[0]}, {upper[1]}, {upper[2]}]")


def main():
    global _clicked_points

    parser = argparse.ArgumentParser(description="Color calibration for HSV range tuning")
    parser.add_argument("--ip", type=str, default=None, help="Robot IP address")
    args = parser.parse_args()

    config = get_config()
    ip = args.ip or config.robot.remote_ip

    # Connect to robot
    print(f"Connecting to robot at {ip}...")
    robot_config = LeKiwiClientConfig(remote_ip=ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to robot!")
        return

    print("Robot connected.")

    # Create output directory
    output_dir = Path("calibration_screenshots")
    output_dir.mkdir(exist_ok=True)

    window_name = "Color Calibration"
    cv2.namedWindow(window_name)

    mode = "click"  # click, pink, black

    # Load and show ROI config
    reload_config()

    print("\n" + "=" * 60)
    print("Color Calibration")
    print("=" * 60)
    print("\nControls:")
    print("  Left click: Show HSV value at clicked point")
    print("  'c': Clear recorded points")
    print("  'p': Show pink detection result")
    print("  'b': Show black detection result")
    print("  'n': Return to normal/click mode")
    print("  'r': Reload config from file")
    print("  's': Save screenshot")
    print("  'q': Quit")
    print("=" * 60)
    print("\nClick on target colors to see their HSV values.")
    print("Multiple clicks will suggest an HSV range.\n")

    try:
        while True:
            obs = robot.get_observation()
            image = obs.get(config.camera.obs_key)

            if image is None:
                print(f"No camera image! (key: {config.camera.obs_key})")
                break

            # Convert to BGR for OpenCV
            image_bgr = convert_to_bgr(image, config.camera.color_format)
            h, w = image_bgr.shape[:2]

            # Set mouse callback with current image
            cv2.setMouseCallback(window_name, mouse_callback, {"image_bgr": image_bgr})

            if mode == "click":
                display = image_bgr.copy()

                # Draw clicked points
                for i, pt in enumerate(_clicked_points[-10:]):  # Show last 10 points
                    px, py = pt["pixel"]
                    hsv = pt["hsv"]
                    sp = pt.get("shoulder_pan", 0.0)
                    pd = pt.get("pixel_distance", 0.0)

                    # Draw circle at clicked point
                    cv2.circle(display, (px, py), 8, (0, 255, 0), 2)
                    cv2.circle(display, (px, py), 2, (0, 255, 0), -1)

                    # Draw HSV and shoulder_pan label
                    label1 = f"H:{hsv[0]} S:{hsv[1]} V:{hsv[2]}"
                    label2 = f"pan:{sp:.1f} dist:{pd:.0f}"
                    cv2.putText(display, label1, (px + 10, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(display, label2, (px + 10, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Show mode and point count
                cv2.putText(display, f"Mode: CLICK ({len(_clicked_points)} points)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, "Click to sample HSV values",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif mode == "pink":
                # Show pink detection
                result = detect_pink_detailed(image_bgr)
                display = image_bgr.copy()

                # Draw ROI if configured
                roi = get_roi(TargetColor.PINK)
                x1, y1, x2, y2 = 0, 0, w, h  # Default to full image
                if roi is not None:
                    rx, ry, rw, rh = roi
                    x1, y1 = int(rx * w), int(ry * h)
                    x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display, "ROI", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Convert to HSV and show mask (only within ROI)
                hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
                pink_cfg = config.color_detection.pink
                lower = np.array(pink_cfg.lower)
                upper = np.array(pink_cfg.upper)
                mask = cv2.inRange(hsv, lower, upper)

                # Apply ROI mask - only show detection within ROI
                roi_mask = np.zeros_like(mask)
                roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                mask = roi_mask

                # Overlay mask in pink color
                mask_colored = np.zeros_like(display)
                mask_colored[:, :, 2] = mask  # Red channel
                mask_colored[:, :, 0] = mask  # Blue channel
                display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

                if result:
                    cx, cy, area, contour = result
                    cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.putText(display, f"Center: ({cx}, {cy}) Area: {int(area)}",
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(display, f"Mode: PINK Detection",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(display, f"Range: {pink_cfg.lower} - {pink_cfg.upper} kernel:{pink_cfg.kernel_size}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                roi_status = f"ROI: {roi}" if roi else "ROI: None (full image)"
                cv2.putText(display, roi_status,
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                cv2.putText(display, f"min_area: {config.color_detection.min_area}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            elif mode == "black":
                # Show black detection
                result = detect_black_detailed(image_bgr)
                display = image_bgr.copy()

                # Draw ROI if configured
                roi = get_roi(TargetColor.BLACK)
                x1, y1, x2, y2 = 0, 0, w, h  # Default to full image
                if roi is not None:
                    rx, ry, rw, rh = roi
                    x1, y1 = int(rx * w), int(ry * h)
                    x2, y2 = int((rx + rw) * w), int((ry + rh) * h)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display, "ROI", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # Convert to HSV and show mask (only within ROI)
                hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
                black_cfg = config.color_detection.black
                lower = np.array(black_cfg.lower)
                upper = np.array(black_cfg.upper)
                mask = cv2.inRange(hsv, lower, upper)

                # Apply ROI mask - only show detection within ROI
                roi_mask = np.zeros_like(mask)
                roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                mask = roi_mask

                # Overlay mask in gray color
                mask_colored = np.zeros_like(display)
                mask_colored[:, :, 0] = mask // 2
                mask_colored[:, :, 1] = mask // 2
                mask_colored[:, :, 2] = mask // 2
                display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

                if result:
                    cx, cy, area, contour = result
                    cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.putText(display, f"Center: ({cx}, {cy}) Area: {int(area)}",
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(display, f"Mode: BLACK Detection",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                cv2.putText(display, f"Range: {black_cfg.lower} - {black_cfg.upper} kernel:{black_cfg.kernel_size}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                roi_status = f"ROI: {roi}" if roi else "ROI: None (full image)"
                cv2.putText(display, roi_status,
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                cv2.putText(display, f"min_area: {config.color_detection.min_area}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                _clicked_points.clear()
                print("\nCleared all recorded points.")
            elif key == ord("p"):
                mode = "pink"
                print("\nSwitched to PINK detection mode")
            elif key == ord("b"):
                mode = "black"
                print("\nSwitched to BLACK detection mode")
            elif key == ord("n"):
                mode = "click"
                print("\nSwitched to CLICK mode")
            elif key == ord("r"):
                reload_config()
                config = get_config()  # Reload local config too
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"color_{mode}_{timestamp}.png"
                cv2.imwrite(str(filename), display)
                print(f"\nScreenshot saved: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cv2.destroyAllWindows()

    # Final summary
    if _clicked_points:
        print("\n" + "=" * 60)
        print("Recorded HSV Values")
        print("=" * 60)
        for i, pt in enumerate(_clicked_points):
            print(f"  {i + 1}. Pixel {pt['pixel']}: HSV {pt['hsv']}")

        h_vals = [p["hsv"][0] for p in _clicked_points]
        s_vals = [p["hsv"][1] for p in _clicked_points]
        v_vals = [p["hsv"][2] for p in _clicked_points]

        margin_h = 10
        margin_s = 30
        margin_v = 30

        lower = [
            max(0, min(h_vals) - margin_h),
            max(0, min(s_vals) - margin_s),
            max(0, min(v_vals) - margin_v),
        ]
        upper = [
            min(180, max(h_vals) + margin_h),
            min(255, max(s_vals) + margin_s),
            min(255, max(v_vals) + margin_v),
        ]

        print(f"\nSuggested HSV range for config.yaml:")
        print(f"  lower: [{lower[0]}, {lower[1]}, {lower[2]}]")
        print(f"  upper: [{upper[0]}, {upper[1]}, {upper[2]}]")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
