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

"""Camera calibration module for pixel to robot control value conversion.

This module converts detected pixel coordinates to:
1. Pixel distance: distance from robot origin to target (in pixels)
2. Shoulder pan: motor value for direction control

The robot origin is below the camera view, so we use the offset to
calculate the correct pixel distance from the virtual origin.
"""

import math

import cv2
import numpy as np

from .config import CalibrationConfig, get_config

# Global calibration instance (loaded from config)
_calibration: CalibrationConfig | None = None


def _get_calibration() -> CalibrationConfig:
    """Get calibration config, loading from file if needed."""
    global _calibration
    if _calibration is None:
        _calibration = get_config().calibration
    return _calibration


def reload_calibration() -> None:
    """Reload calibration from config file."""
    global _calibration
    _calibration = get_config().calibration


def pixel_to_robot(px: int, py: int, image_height: int = 480) -> tuple[float, float]:
    """
    Convert image coordinates to pixel distance and shoulder_pan value.

    Args:
        px: Image X coordinate (pixel)
        py: Image Y coordinate (pixel)
        image_height: Height of the image in pixels (default 480)

    Returns:
        (pixel_distance, shoulder_pan) tuple
        - pixel_distance: Distance from robot origin to target in pixels
        - shoulder_pan: Interpolated shoulder_pan motor value
    """
    cal = _get_calibration()

    # === Pixel distance calculation ===
    # Virtual robot origin is at (center_x, image_height + offset)
    origin_x = cal.image_center_x
    origin_y = image_height + cal.robot_origin_offset_px

    # Calculate pixel distance from origin to target
    dx = px - origin_x
    dy = origin_y - py  # Positive because y increases downward in image
    pixel_distance = math.sqrt(dx * dx + dy * dy)

    # === Shoulder pan interpolation based on angle ===
    # Calculate the angle from robot origin to target point
    target_angle = math.degrees(math.atan2(dx, dy)) if dy > 0 else 0.0

    # Calculate angles for calibration points (at reference_y)
    ref_y = cal.direction_reference_y
    ref_dy = origin_y - ref_y

    direction_points = sorted(cal.direction_points, key=lambda p: p.pixel_x)

    if len(direction_points) == 0:
        shoulder_pan = 0.0
    elif len(direction_points) == 1:
        shoulder_pan = direction_points[0].shoulder_pan
    else:
        # Build angle -> shoulder_pan mapping from calibration points
        cal_angles = []
        for dp in direction_points:
            cal_dx = dp.pixel_x - origin_x
            cal_angle = math.degrees(math.atan2(cal_dx, ref_dy)) if ref_dy > 0 else 0.0
            cal_angles.append((cal_angle, dp.shoulder_pan))

        # Sort by angle
        cal_angles.sort(key=lambda x: x[0])

        # Interpolate/extrapolate based on angle
        if target_angle <= cal_angles[0][0]:
            # Extrapolate using first two points
            a1, sp1 = cal_angles[0]
            a2, sp2 = cal_angles[1]
            slope = (sp2 - sp1) / (a2 - a1) if a2 != a1 else 0.0
            shoulder_pan = sp1 + slope * (target_angle - a1)
        elif target_angle >= cal_angles[-1][0]:
            # Extrapolate using last two points
            a1, sp1 = cal_angles[-2]
            a2, sp2 = cal_angles[-1]
            slope = (sp2 - sp1) / (a2 - a1) if a2 != a1 else 0.0
            shoulder_pan = sp2 + slope * (target_angle - a2)
        else:
            # Find the two angles to interpolate between
            for i in range(len(cal_angles) - 1):
                a1, sp1 = cal_angles[i]
                a2, sp2 = cal_angles[i + 1]
                if a1 <= target_angle <= a2:
                    t = (target_angle - a1) / (a2 - a1)
                    shoulder_pan = sp1 + t * (sp2 - sp1)
                    break
            else:
                shoulder_pan = 0.0

    return (pixel_distance, shoulder_pan)


def draw_calibration_overlay(image: np.ndarray) -> np.ndarray:
    """
    Draw calibration visualization overlay on the image.

    Shows:
    - Robot origin position (virtual, below image)
    - Center line from robot origin
    - Direction calibration points

    Args:
        image: BGR image (numpy array)

    Returns:
        Image with calibration overlay drawn
    """
    cal = _get_calibration()
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    origin_x = cal.image_center_x
    ref_y = cal.direction_reference_y

    # Draw center vertical line (from top to bottom of image)
    cv2.line(img_copy, (origin_x, 0), (origin_x, h), (255, 255, 0), 1)

    # Draw reference Y horizontal line
    cv2.line(img_copy, (0, ref_y), (w, ref_y), (0, 255, 0), 1)
    cv2.putText(img_copy, f"ref_y={ref_y}", (w - 80, ref_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw direction calibration points
    for dp in cal.direction_points:
        # Draw line from origin to calibration point
        color = (0, 255, 255) if dp.pixel_x < origin_x else (255, 0, 255)
        if dp.pixel_x == origin_x:
            color = (0, 255, 0)
        cv2.line(img_copy, (origin_x, h), (dp.pixel_x, ref_y), color, 1)

        # Draw calibration point
        cv2.circle(img_copy, (dp.pixel_x, ref_y), 6, color, -1)

        # Draw label
        label = f"pan={dp.shoulder_pan:.1f}"
        cv2.putText(img_copy, label, (dp.pixel_x + 5, ref_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw robot origin indicator (at bottom of image)
    cv2.circle(img_copy, (origin_x, h - 5), 8, (0, 0, 255), -1)
    cv2.putText(img_copy, f"Origin: +{cal.robot_origin_offset_px}px below",
                (origin_x - 80, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return img_copy


def draw_calibration_points(image: np.ndarray) -> np.ndarray:
    """Draw calibration reference points on the image (alias for compatibility)."""
    return draw_calibration_overlay(image)


def visualize_detection(
    image: np.ndarray, cx: int, cy: int, pixel_distance: float, shoulder_pan: float
) -> np.ndarray:
    """
    Visualize detected object position with calculated values.

    Args:
        image: BGR image (numpy array)
        cx, cy: Detected center point coordinates
        pixel_distance: Calculated pixel distance from origin
        shoulder_pan: Calculated shoulder_pan value

    Returns:
        Visualized image
    """
    cal = _get_calibration()
    img_copy = draw_calibration_overlay(image)
    h, w = img_copy.shape[:2]

    origin_x = cal.image_center_x

    # Draw detected point
    cv2.circle(img_copy, (cx, cy), 12, (255, 0, 255), -1)  # Magenta filled
    cv2.circle(img_copy, (cx, cy), 14, (255, 255, 255), 2)  # White outline

    # Draw line from origin to detected point
    cv2.line(img_copy, (origin_x, h), (cx, cy), (255, 0, 255), 2)

    # Draw info text at top
    info = f"Pixel dist: {pixel_distance:.0f}px, Shoulder pan: {shoulder_pan:.2f}"
    cv2.putText(img_copy, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw coordinates
    coord_info = f"Pixel: ({cx}, {cy})"
    cv2.putText(img_copy, coord_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_copy


def get_detection_info(px: int, py: int, image_height: int = 480) -> dict:
    """
    Get complete detection information from pixel coordinates.

    Args:
        px: Image X coordinate
        py: Image Y coordinate
        image_height: Height of the image

    Returns:
        Dictionary with pixel coords, pixel distance, and shoulder_pan
    """
    pixel_distance, shoulder_pan = pixel_to_robot(px, py, image_height)

    return {
        "pixel": (px, py),
        "pixel_distance": pixel_distance,
        "shoulder_pan": shoulder_pan,
    }


# Legacy function aliases for compatibility
def pixel_to_real(px: int, py: int, image_height: int = 480) -> tuple[float, float]:
    """Legacy alias for pixel_to_robot. Returns (pixel_distance, shoulder_pan)."""
    return pixel_to_robot(px, py, image_height)


def real_to_normalized(pixel_distance: float, shoulder_pan: float) -> tuple[float, float]:
    """
    Normalize pixel distance for trajectory playback.

    Note: shoulder_pan is returned as-is since it's already a motor value.

    Args:
        pixel_distance: Distance in pixels
        shoulder_pan: Shoulder pan motor value

    Returns:
        (normalized_distance, shoulder_pan) tuple
        - normalized_distance: 0.0 to 1.0 based on typical range
        - shoulder_pan: Unchanged motor value
    """
    # Normalize distance assuming typical range of 100-500 pixels
    # Adjust these values based on your actual working range
    min_dist = 100.0
    max_dist = 500.0
    normalized_distance = (pixel_distance - min_dist) / (max_dist - min_dist)
    normalized_distance = max(0.0, min(1.0, normalized_distance))

    return (normalized_distance, shoulder_pan)
