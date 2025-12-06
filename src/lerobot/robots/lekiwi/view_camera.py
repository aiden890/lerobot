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
View camera images from LeKiwi robot in real-time with Hough transform line detection.

Usage:
    python -m lerobot.robots.lekiwi.view_camera
"""

import logging
import time

import cv2
import numpy as np

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.robots.lekiwi import line_detection

# ============================================================================
# 파라미터 설정 (Parameter Configuration)
# ============================================================================

# 로봇 연결 설정
ROBOT_IP = "192.168.20.199"  # LeKiwi 로봇 IP 주소
CAMERA_FPS = 10              # 카메라 프레임 속도 (Hz)

# ============================================================================
# 라인 detection 파라미터는 line_detection.py에서 설정
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Global detector instance with smoothing
_line_detector = None


def apply_parallel_line_detection(image: np.ndarray, fps: float = 0.0) -> np.ndarray:
    """
    Apply parallel line detection to the image and show all detection stages.

    Args:
        image: Input BGR image
        fps: Current frames per second to display

    Returns:
        Image with:
        - Raw Hough lines (thin green, thickness=1)
        - Center lines from parallel pairs (medium red, thickness=3)
        - Vertical polylines after merging (thick blue, thickness=5)
        - Horizontal polylines after merging (thick orange, thickness=5)
        - Diagonal polylines after merging (thick green, thickness=5)
        - Junction points (magenta/cyan/yellow circles)
    """
    global _line_detector
    if _line_detector is None:
        _line_detector = line_detection.SmoothedLineDetector()

    # Create a copy for drawing
    result = image.copy()
    height, width = image.shape[:2]

    # Detect raw Hough lines
    raw_lines, roi_bounds = line_detection.detect_lines_in_roi(image)

    # Draw ROI boundary for reference
    roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_bounds
    cv2.rectangle(result, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

    # Draw raw Hough lines in THIN GREEN
    if raw_lines is not None and len(raw_lines) > 0:
        for line in raw_lines:
            x1, y1, x2, y2 = line[:4]
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    # Get parallel pairs and calculate center lines
    parallel_pairs = line_detection.select_all_parallel_line_pairs(raw_lines, width, height)
    center_lines = []
    if parallel_pairs:
        for line1, line2 in parallel_pairs:
            center_line = line_detection.calculate_center_line(line1, line2)
            center_lines.append(center_line)

    # Draw center lines in RED (medium thickness)
    for center_line in center_lines:
        x1, y1, x2, y2 = center_line[:4]
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    # Detect lines with smoothing (parallel pairs + merging)
    vertical_polylines, horizontal_polylines, diagonal_polylines = _line_detector.detect_and_smooth(image)

    # Draw vertical polylines in BLUE
    for polyline in vertical_polylines:
        for i in range(len(polyline) - 1):
            pt1 = tuple(map(int, polyline[i]))
            pt2 = tuple(map(int, polyline[i + 1]))
            cv2.line(result, pt1, pt2, (255, 0, 0), 5)

        # Draw junction points in magenta
        for point in polyline:
            cv2.circle(result, tuple(map(int, point)), 5, (255, 0, 255), -1)

    # Draw horizontal polylines in ORANGE
    for polyline in horizontal_polylines:
        for i in range(len(polyline) - 1):
            pt1 = tuple(map(int, polyline[i]))
            pt2 = tuple(map(int, polyline[i + 1]))
            cv2.line(result, pt1, pt2, (0, 165, 255), 5)

        # Draw junction points in cyan
        for point in polyline:
            cv2.circle(result, tuple(map(int, point)), 5, (255, 255, 0), -1)

    # Draw diagonal polylines in GREEN
    for polyline in diagonal_polylines:
        for i in range(len(polyline) - 1):
            pt1 = tuple(map(int, polyline[i]))
            pt2 = tuple(map(int, polyline[i + 1]))
            cv2.line(result, pt1, pt2, (0, 255, 0), 5)

        # Draw junction points in yellow
        for point in polyline:
            cv2.circle(result, tuple(map(int, point)), 5, (0, 255, 255), -1)

    # Draw 100 pixel scale reference at bottom-left
    scale_x = 20
    scale_y = height - 40
    scale_length = 100
    cv2.line(result, (scale_x, scale_y), (scale_x + scale_length, scale_y), (255, 255, 255), 2)
    cv2.line(result, (scale_x, scale_y - 5), (scale_x, scale_y + 5), (255, 255, 255), 2)
    cv2.line(result, (scale_x + scale_length, scale_y - 5), (scale_x + scale_length, scale_y + 5), (255, 255, 255), 2)
    cv2.putText(result, "100 px", (scale_x + 30, scale_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add status text with FPS and smoothing status
    total_polylines = len(vertical_polylines) + len(horizontal_polylines) + len(diagonal_polylines)
    raw_line_count = len(raw_lines) if raw_lines is not None else 0
    center_line_count = len(center_lines)
    smoothing_text = f"[S:{_line_detector.detection_count}/{_line_detector.min_confidence}]"

    if total_polylines > 0:
        status_text = f"Hough:{raw_line_count} Center:{center_line_count} V:{len(vertical_polylines)} H:{len(horizontal_polylines)} D:{len(diagonal_polylines)} {smoothing_text} | {fps:.1f} Hz"
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        status_text = f"Hough:{raw_line_count} Center:{center_line_count} NO LINES {smoothing_text} | {fps:.1f} Hz"
        cv2.putText(result, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return result


def main():

    # Robot configuration
    robot_config = LeKiwiClientConfig(remote_ip=ROBOT_IP, id="lekiwi")
    robot = LeKiwiClient(robot_config)

    # Connect to robot
    logger.info("Connecting to LeKiwi...")
    robot.connect()
    logger.info("Connected!")

    # Camera viewer settings
    fps = CAMERA_FPS
    dt = 1.0 / fps

    try:
        logger.info(f"Starting camera viewer with parallel line detection at {fps}Hz. Press 'q' to quit.")

        while True:
            start_time = time.perf_counter()

            # Get observation from robot
            observation = robot.get_observation()

            # Calculate actual FPS
            elapsed = time.perf_counter() - start_time
            actual_fps = 1.0 / max(elapsed, 1e-6)

            # Display camera images with line detection
            for camera_name in robot.config.cameras.keys():
                if camera_name in observation:
                    image = observation[camera_name]
                    detected_image = apply_parallel_line_detection(image, actual_fps)
                    cv2.imshow(f"LeKiwi - {camera_name} (Line Detection)", detected_image)

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit signal received")
                break

            # Maintain target Hz loop rate
            elapsed = time.perf_counter() - start_time
            sleep_time = max(dt - elapsed, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Log actual FPS occasionally
            if int(start_time * 10) % 30 == 0:  # Log every ~3 seconds
                logger.debug(f"Running at {actual_fps:.1f} FPS")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        robot.disconnect()
        logger.info("Disconnected from robot")


if __name__ == "__main__":
    main()
