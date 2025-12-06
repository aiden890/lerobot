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

"""Color detection module for pink and black region detection."""

from enum import Enum

import cv2
import numpy as np


def convert_to_bgr(image: np.ndarray, color_format: str = "rgb") -> np.ndarray:
    """
    Convert image to BGR format for OpenCV processing.

    Args:
        image: Input image (numpy array)
        color_format: Input format - "rgb" or "bgr"

    Returns:
        BGR image
    """
    if color_format.lower() == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image  # Already BGR


class TargetColor(Enum):
    """Target colors for detection."""

    PINK = "pink"  # For pick target (object to grasp)
    BLACK = "black"  # For place target (destination)


# HSV color ranges - loaded from config, with fallback defaults
COLOR_RANGES: dict[TargetColor, dict] = {
    TargetColor.PINK: {
        "lower": np.array([140, 50, 50]),  # Default fallback
        "upper": np.array([170, 255, 255]),
        "kernel_size": 5,
    },
    TargetColor.BLACK: {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([180, 255, 50]),
        "kernel_size": 5,
    },
}

# Flag to track if config has been loaded
_config_loaded = False

# ROI configuration (loaded from config)
ROI_CONFIG: dict[TargetColor, list[float] | None] = {
    TargetColor.PINK: None,
    TargetColor.BLACK: None,
}

# Minimum contour area (loaded from config)
MIN_AREA: int = 400  # Default fallback


def _load_color_ranges_from_config() -> None:
    """Load color ranges and ROI from config.yaml."""
    global _config_loaded, MIN_AREA
    if _config_loaded:
        return

    try:
        from .config import get_config

        config = get_config()

        COLOR_RANGES[TargetColor.PINK]["lower"] = np.array(config.color_detection.pink.lower)
        COLOR_RANGES[TargetColor.PINK]["upper"] = np.array(config.color_detection.pink.upper)
        COLOR_RANGES[TargetColor.PINK]["kernel_size"] = config.color_detection.pink.kernel_size
        COLOR_RANGES[TargetColor.BLACK]["lower"] = np.array(config.color_detection.black.lower)
        COLOR_RANGES[TargetColor.BLACK]["upper"] = np.array(config.color_detection.black.upper)
        COLOR_RANGES[TargetColor.BLACK]["kernel_size"] = config.color_detection.black.kernel_size

        # Load ROI config
        ROI_CONFIG[TargetColor.PINK] = config.roi.pink
        ROI_CONFIG[TargetColor.BLACK] = config.roi.black

        # Load min_area from config
        MIN_AREA = config.color_detection.min_area

        _config_loaded = True
    except Exception as e:
        print(f"Warning: Failed to load color detection config: {e}")
        _config_loaded = True  # Don't retry, use defaults


def _apply_roi(
    image: np.ndarray, roi: list[float] | None
) -> tuple[np.ndarray, int, int]:
    """
    Apply ROI to image.

    Args:
        image: Input image
        roi: ROI as [x, y, width, height] in normalized coordinates (0.0-1.0), or None for full image

    Returns:
        (cropped_image, x_offset, y_offset) - cropped image and pixel offsets for coordinate conversion
    """
    if roi is None:
        return image, 0, 0

    height, width = image.shape[:2]
    x, y, w, h = roi

    # Convert normalized coordinates to pixel coordinates
    x1 = int(x * width)
    y1 = int(y * height)
    x2 = int((x + w) * width)
    y2 = int((y + h) * height)

    # Clamp to image bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return image[y1:y2, x1:x2], x1, y1


def detect_color_region(
    image: np.ndarray,
    color: TargetColor,
    min_area: int | None = None,
    roi: list[float] | None = None,
) -> tuple[int, int, float, float] | None:
    """
    Detect a colored region and return its center point with normalized distance and angle.

    Args:
        image: BGR image (numpy array)
        color: Target color to detect (PINK or BLACK)
        min_area: Minimum contour area in pixels (uses config value if None)
        roi: Optional ROI override [x, y, width, height] in normalized coords (0.0-1.0)
             If None, uses ROI from config. Pass empty list [] to disable ROI.

    Returns:
        (center_x, center_y, normalized_distance, normalized_angle) or None if not found
        - center_x, center_y: Pixel coordinates of detected region center (in original image coords)
        - normalized_distance: 0.0 (bottom, near) to 1.0 (top, far)
        - normalized_angle: -1.0 (left) to 0.0 (center) to 1.0 (right)
    """
    # Load color ranges from config on first use
    _load_color_ranges_from_config()

    # Use config min_area if not specified
    if min_area is None:
        min_area = MIN_AREA

    # Get original image dimensions for normalized calculations
    orig_height, orig_width = image.shape[:2]

    # Apply ROI (use config ROI if not explicitly provided)
    if roi is None:
        roi = ROI_CONFIG.get(color)
    elif roi == []:
        roi = None  # Empty list means no ROI

    roi_image, x_offset, y_offset = _apply_roi(image, roi)

    # Check if ROI is valid
    if roi_image.size == 0:
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    # Get color range
    range_info = COLOR_RANGES[color]
    mask = cv2.inRange(hsv, range_info["lower"], range_info["upper"])

    # Morphological operations to reduce noise (kernel size from config)
    kernel_size = range_info.get("kernel_size", 5)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Select largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    # Calculate center using moments
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    # Get center in ROI coordinates
    cx_roi = int(M["m10"] / M["m00"])
    cy_roi = int(M["m01"] / M["m00"])

    # Convert to original image coordinates
    cx = cx_roi + x_offset
    cy = cy_roi + y_offset

    # Calculate normalized distance (Y coordinate based, using original image dimensions)
    # Bottom of image (high Y) = near = 0.0
    # Top of image (low Y) = far = 1.0
    normalized_distance = 1.0 - (cy / orig_height)

    # Calculate normalized angle (X coordinate based, using original image dimensions)
    # Left side = -1.0, Center = 0.0, Right side = 1.0
    normalized_angle = (cx - orig_width / 2) / (orig_width / 2)

    return (cx, cy, normalized_distance, normalized_angle)


def detect_pink(
    image: np.ndarray, min_area: int | None = None, roi: list[float] | None = None
) -> tuple[int, int, float, float] | None:
    """
    Detect pink object for pick operation.

    Args:
        image: BGR image
        min_area: Minimum contour area (uses config value if None)
        roi: Optional ROI override [x, y, width, height] in normalized coords (0.0-1.0)

    Returns:
        (cx, cy, normalized_distance, normalized_angle) or None
    """
    return detect_color_region(image, TargetColor.PINK, min_area, roi)


def detect_black(
    image: np.ndarray, min_area: int | None = None, roi: list[float] | None = None
) -> tuple[int, int, float, float] | None:
    """
    Detect black region for place operation.

    Args:
        image: BGR image
        min_area: Minimum contour area (uses config value if None)
        roi: Optional ROI override [x, y, width, height] in normalized coords (0.0-1.0)

    Returns:
        (cx, cy, normalized_distance, normalized_angle) or None
    """
    return detect_color_region(image, TargetColor.BLACK, min_area, roi)


def get_color_mask(image: np.ndarray, color: TargetColor) -> np.ndarray:
    """
    Get the binary mask for a specific color.

    Useful for debugging and visualization.

    Args:
        image: BGR image
        color: Target color

    Returns:
        Binary mask (numpy array)
    """
    _load_color_ranges_from_config()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    range_info = COLOR_RANGES[color]
    mask = cv2.inRange(hsv, range_info["lower"], range_info["upper"])

    kernel_size = range_info.get("kernel_size", 5)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def visualize_color_detection(
    image: np.ndarray,
    color: TargetColor,
    min_area: int | None = None,
    show_roi: bool = True,
) -> np.ndarray:
    """
    Visualize color detection result on the image.

    Args:
        image: BGR image
        color: Target color
        min_area: Minimum contour area (uses config value if None)
        show_roi: Whether to draw ROI rectangle

    Returns:
        Annotated image showing detection result
    """
    # Load config to get ROI
    _load_color_ranges_from_config()

    # Use config min_area if not specified
    if min_area is None:
        min_area = MIN_AREA

    img_copy = image.copy()
    height, width = image.shape[:2]

    # Draw ROI rectangle if configured
    roi = ROI_CONFIG.get(color)
    if show_roi and roi is not None:
        x, y, w, h = roi
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + w) * width)
        y2 = int((y + h) * height)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta ROI box
        cv2.putText(img_copy, "ROI", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    result = detect_color_region(image, color, min_area)

    # Get and overlay mask (only within ROI for visualization)
    mask = get_color_mask(image, color)
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    colored_mask[:, :, 0] = 0  # Remove blue channel
    colored_mask[:, :, 2] = 0  # Remove red channel for green overlay
    img_copy = cv2.addWeighted(img_copy, 0.7, colored_mask, 0.3, 0)

    if result:
        cx, cy, norm_dist, norm_angle = result
        # Draw center point
        cv2.circle(img_copy, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(img_copy, (cx, cy), 12, (255, 255, 255), 2)

        # Draw info text
        info = f"{color.value}: ({cx}, {cy}) dist={norm_dist:.2f} angle={norm_angle:.2f}"
        cv2.putText(img_copy, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(img_copy, f"{color.value}: Not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img_copy


def update_color_range(color: TargetColor, lower: list[int], upper: list[int]) -> None:
    """
    Update HSV color range for a target color.

    Useful for runtime calibration.

    Args:
        color: Target color to update
        lower: Lower HSV bound [H, S, V]
        upper: Upper HSV bound [H, S, V]
    """
    COLOR_RANGES[color]["lower"] = np.array(lower)
    COLOR_RANGES[color]["upper"] = np.array(upper)


def update_roi(color: TargetColor, roi: list[float] | None) -> None:
    """
    Update ROI for a target color.

    Useful for runtime calibration.

    Args:
        color: Target color to update
        roi: ROI as [x, y, width, height] in normalized coords (0.0-1.0), or None to disable
    """
    ROI_CONFIG[color] = roi


def get_roi(color: TargetColor) -> list[float] | None:
    """
    Get current ROI for a target color.

    Args:
        color: Target color

    Returns:
        ROI as [x, y, width, height] in normalized coords, or None if not set
    """
    _load_color_ranges_from_config()
    return ROI_CONFIG.get(color)


def reload_config() -> None:
    """Force reload configuration from config.yaml."""
    global _config_loaded
    _config_loaded = False
    _load_color_ranges_from_config()
    print(f"Config reloaded.")
    print(f"  min_area: {MIN_AREA}")
    print(f"  ROI pink: {ROI_CONFIG.get(TargetColor.PINK)}")
    print(f"  ROI black: {ROI_CONFIG.get(TargetColor.BLACK)}")
    print(f"  black kernel_size: {COLOR_RANGES[TargetColor.BLACK].get('kernel_size', 5)}")


def detect_color_region_detailed(
    image: np.ndarray,
    color: TargetColor,
    min_area: int | None = None,
    roi: list[float] | None = None,
) -> tuple[int, int, float, np.ndarray] | None:
    """
    Detect a colored region and return detailed info including contour (for calibration/debug).

    Args:
        image: BGR image (numpy array)
        color: Target color to detect (PINK or BLACK)
        min_area: Minimum contour area in pixels (uses config value if None)
        roi: Optional ROI override [x, y, width, height] in normalized coords (0.0-1.0)

    Returns:
        (center_x, center_y, area, contour) or None if not found
        - center_x, center_y: Pixel coordinates in original image
        - area: Contour area in pixels
        - contour: Contour points (offset to original image coordinates)
    """
    _load_color_ranges_from_config()

    # Use config min_area if not specified
    if min_area is None:
        min_area = MIN_AREA

    # Apply ROI
    if roi is None:
        roi = ROI_CONFIG.get(color)
    elif roi == []:
        roi = None

    roi_image, x_offset, y_offset = _apply_roi(image, roi)

    if roi_image.size == 0:
        return None

    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    range_info = COLOR_RANGES[color]
    mask = cv2.inRange(hsv, range_info["lower"], range_info["upper"])

    kernel_size = range_info.get("kernel_size", 5)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx_roi = int(M["m10"] / M["m00"])
    cy_roi = int(M["m01"] / M["m00"])

    # Convert to original image coordinates
    cx = cx_roi + x_offset
    cy = cy_roi + y_offset

    # Offset contour points to original image coordinates
    contour_offset = largest.copy()
    contour_offset[:, :, 0] += x_offset
    contour_offset[:, :, 1] += y_offset

    return (cx, cy, area, contour_offset)


def detect_pink_detailed(
    image: np.ndarray, min_area: int | None = None, roi: list[float] | None = None
) -> tuple[int, int, float, np.ndarray] | None:
    """Detect pink with detailed info (for calibration/debug)."""
    return detect_color_region_detailed(image, TargetColor.PINK, min_area, roi)


def detect_black_detailed(
    image: np.ndarray, min_area: int | None = None, roi: list[float] | None = None
) -> tuple[int, int, float, np.ndarray] | None:
    """Detect black with detailed info (for calibration/debug)."""
    return detect_color_region_detailed(image, TargetColor.BLACK, min_area, roi)
