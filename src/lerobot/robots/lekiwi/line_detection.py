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
Common line detection utilities for LeKiwi using Hough transform.
"""

import math

import cv2
import numpy as np


# ============================================================================
# 파라미터 설정 (Parameter Configuration)
# ============================================================================

# ROI (Region of Interest) 설정 - 선을 검출할 이미지 영역
ROI_Y_TOP = 0.2             # 세로 시작 위치 (0.0 = 상단, 1.0 = 하단)
ROI_Y_BOTTOM = 0.6          # 세로 끝 위치 (0.0 = 상단, 1.0 = 하단) - 상단 70%
ROI_X_LEFT = 0.2           # 가로 시작 위치 (0.0 = 좌측, 1.0 = 우측)
ROI_X_RIGHT = 0.8          # 가로 끝 위치 (0.0 = 좌측, 1.0 = 우측) - 중앙 50%

# 허프 변환 파라미터 (Hough Transform Parameters)
HOUGH_THRESHOLD = 40        # 직선 검출 투표 수 (높을수록 더 명확한 선만 검출)
MIN_LINE_LENGTH = 60       # 최소 직선 길이 (픽셀)
MAX_LINE_GAP = 20           # 선분 간 최대 간격 (픽셀)

# Canny 엣지 검출 파라미터
CANNY_THRESHOLD1 = 50       # Canny 하위 임계값
CANNY_THRESHOLD2 = 150      # Canny 상위 임계값

# 평행선 탐지 파라미터 (Parallel Line Detection Parameters)
PARALLEL_ANGLE_THRESHOLD = 10.0     # 평행으로 간주할 최대 각도 차이 (도)
MIN_LINE_DISTANCE = 6               # 두 라인 간 최소 거리 (픽셀)
MAX_LINE_DISTANCE = 30              # 두 라인 간 최대 거리 (픽셀)

# 선분 병합 파라미터 (Line Merging Parameters)
ENDPOINT_DISTANCE_THRESHOLD = 60    # 끝점 연결 판정 거리 (픽셀)

# 선 방향 분류 파라미터 (Line Direction Classification Parameters)
VERTICAL_ANGLE_MAX = 20.0          # 세로선 최대 각도 (0도 = 완전 수직)
HORIZONTAL_ANGLE_MIN = 70.0        # 가로선 최소 각도 (90도 = 완전 수평)
# 대각선: VERTICAL_ANGLE_MAX < angle < HORIZONTAL_ANGLE_MIN

# Temporal smoothing 파라미터
SMOOTHING_ALPHA = 0.2               # EMA smoothing factor (0=완전 smoothing, 1=smoothing 없음)
MIN_DETECTION_CONFIDENCE = 10       # 최소 detection 신뢰도 (연속 N 프레임)

# ============================================================================


def detect_lines_in_roi(
    image: np.ndarray,
    roi_y_top: float = ROI_Y_TOP,
    roi_y_bottom: float = ROI_Y_BOTTOM,
    roi_x_left: float = ROI_X_LEFT,
    roi_x_right: float = ROI_X_RIGHT,
    hough_threshold: int = HOUGH_THRESHOLD,
    min_line_length: int = MIN_LINE_LENGTH,
    max_line_gap: int = MAX_LINE_GAP,
    canny_threshold1: int = CANNY_THRESHOLD1,
    canny_threshold2: int = CANNY_THRESHOLD2,
) -> tuple[list, tuple[int, int, int, int]]:
    """
    Detect lines in image ROI using Hough transform.

    Args:
        image: Input BGR image
        roi_y_top: ROI top position (0.0 = top, 1.0 = bottom)
        roi_y_bottom: ROI bottom position
        roi_x_left: ROI left position (0.0 = left, 1.0 = right)
        roi_x_right: ROI right position
        hough_threshold: Minimum votes for line detection
        min_line_length: Minimum line length in pixels
        max_line_gap: Maximum gap between line segments
        canny_threshold1: Canny lower threshold
        canny_threshold2: Canny upper threshold

    Returns:
        Tuple of (list of detected lines in full image coordinates, ROI bounds (x_start, y_start, x_end, y_end))
    """
    height, width = image.shape[:2]

    # Define ROI
    roi_y_start = int(height * roi_y_top)
    roi_y_end = int(height * roi_y_bottom)
    roi_x_start = int(width * roi_x_left)
    roi_x_end = int(width * roi_x_right)

    # Extract ROI
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Adjust line coordinates back to full image
    adjusted_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Adjust coordinates back to full image
            x1_adj = x1 + roi_x_start
            y1_adj = y1 + roi_y_start
            x2_adj = x2 + roi_x_start
            y2_adj = y2 + roi_y_start
            adjusted_lines.append([x1_adj, y1_adj, x2_adj, y2_adj])

    roi_bounds = (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    return adjusted_lines, roi_bounds


def calculate_line_angle(line: list | np.ndarray) -> float:
    """
    Calculate the angle of the line relative to vertical.

    Args:
        line: Line as [x1, y1, x2, y2]

    Returns:
        Angle in radians (-pi/2 to pi/2)
    """
    x1, y1, x2, y2 = line[:4] if isinstance(line, np.ndarray) else line

    # Calculate angle (note: y increases downward in image coordinates)
    dx = x2 - x1
    dy = y2 - y1

    # Angle from horizontal
    angle = math.atan2(dy, dx)

    # Convert to angle from vertical (robot's forward direction)
    # 0 means line is vertical (straight ahead)
    # Positive means line goes to the right
    # Negative means line goes to the left
    angle_from_vertical = angle - math.pi/2

    return angle_from_vertical


def calculate_line_distance(line1: list, line2: list) -> float:
    """
    Calculate perpendicular distance between two parallel lines.

    Uses the formula for distance from a point to a line:
    d = |ax0 + by0 + c| / sqrt(a^2 + b^2)

    Args:
        line1: First line as [x1, y1, x2, y2]
        line2: Second line as [x1, y1, x2, y2]

    Returns:
        Perpendicular distance between the two lines in pixels
    """
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2

    # Convert line1 to line equation: ax + by + c = 0
    # Direction vector of line1
    dx = x2_1 - x1_1
    dy = y2_1 - y1_1

    # Normal vector (perpendicular to line direction)
    # For line through (x1,y1) with direction (dx,dy), normal is (-dy, dx)
    a = -dy
    b = dx
    c = -(a * x1_1 + b * y1_1)

    # Normalize the line equation (so a^2 + b^2 = 1)
    norm = math.sqrt(a * a + b * b)
    if norm < 1e-6:
        # Degenerate line, use center distance as fallback
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2
        return math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

    a /= norm
    b /= norm
    c /= norm

    # Calculate perpendicular distance from both endpoints of line2 to line1
    # Distance from point (x0, y0) to line ax + by + c = 0 is |ax0 + by0 + c|
    dist1 = abs(a * x1_2 + b * y1_2 + c)
    dist2 = abs(a * x2_2 + b * y2_2 + c)

    # Return average distance (both should be similar for parallel lines)
    distance = (dist1 + dist2) / 2

    return distance


def select_all_parallel_line_pairs(
    lines: list,
    image_width: int,
    image_height: int,
    angle_threshold: float = PARALLEL_ANGLE_THRESHOLD,
    min_distance: float = MIN_LINE_DISTANCE,
    max_distance: float = MAX_LINE_DISTANCE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Select ALL pairs of parallel lines from detected lines (not just the best one).

    Returns:
        List of tuples (line1, line2) as [x1, y1, x2, y2] arrays
    """
    if len(lines) < 2:
        return []

    parallel_pairs = []

    # Compare all pairs of lines
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]

            # Calculate angles
            angle1 = calculate_line_angle(line1)
            angle2 = calculate_line_angle(line2)
            angle_diff = abs(math.degrees(angle1 - angle2))

            # Normalize angle difference to [0, 90] range
            # Lines at 5° and 175° are actually parallel (difference should be 10°, not 170°)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            # Check if lines are parallel
            if angle_diff > angle_threshold:
                continue

            # Calculate distance between lines
            distance = calculate_line_distance(line1, line2)

            # Check if distance is within acceptable range
            if distance < min_distance or distance > max_distance:
                continue

            # Add this pair
            parallel_pairs.append((np.array(line1[:4]), np.array(line2[:4])))

    return parallel_pairs


def calculate_center_line(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Calculate the center line between two parallel lines.

    Args:
        line1: First line as [x1, y1, x2, y2]
        line2: Second line as [x1, y1, x2, y2]

    Returns:
        Center line as [x1, y1, x2, y2]
    """
    def _order_line(line: np.ndarray) -> np.ndarray:
        """Ensure lines are ordered from top to bottom to avoid diagonal averages."""
        x1, y1, x2, y2 = line
        if y1 > y2:
            return np.array([x2, y2, x1, y1])
        return np.array([x1, y1, x2, y2])

    ordered_line1 = _order_line(np.asarray(line1, dtype=float))
    ordered_line2 = _order_line(np.asarray(line2, dtype=float))

    x1_1, y1_1, x2_1, y2_1 = ordered_line1
    x1_2, y1_2, x2_2, y2_2 = ordered_line2

    # Calculate center points
    cx1 = (x1_1 + x1_2) / 2
    cy1 = (y1_1 + y1_2) / 2
    cx2 = (x2_1 + x2_2) / 2
    cy2 = (y2_1 + y2_2) / 2

    return np.array([cx1, cy1, cx2, cy2])


def classify_line_direction(
    line: np.ndarray,
    vertical_max: float = VERTICAL_ANGLE_MAX,
    horizontal_min: float = HORIZONTAL_ANGLE_MIN
) -> str:
    """
    선분의 방향을 세로선, 가로선, 대각선으로 분류한다.

    Args:
        line: [x1, y1, x2, y2] 형식의 선분
        vertical_max: 세로선 최대 각도 (도)
        horizontal_min: 가로선 최소 각도 (도)

    Returns:
        "vertical", "horizontal", "diagonal" 중 하나
    """
    x1, y1, x2, y2 = line[:4]
    dx = x2 - x1
    dy = y2 - y1

    # 수직으로부터의 각도 계산 (0도 = 수직, 90도 = 수평)
    angle = abs(math.degrees(math.atan2(abs(dx), abs(dy))))

    if angle <= vertical_max:
        return "vertical"
    elif angle >= horizontal_min:
        return "horizontal"
    else:
        return "diagonal"


def merge_connected_lines_by_direction(
    lines: list,
    distance_threshold: float = ENDPOINT_DISTANCE_THRESHOLD
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """
    끝점이 가까운 선분들을 방향별로 분리하여 병합한다.
    세로선, 가로선, 대각선끼리 각각 병합한다.

    Args:
        lines: 검출된 선분들 [[x1, y1, x2, y2], ...]
        distance_threshold: 끝점 사이의 최대 연결 거리 (픽셀)

    Returns:
        (vertical_polylines, horizontal_polylines, diagonal_polylines)
        각 폴리라인은 점들의 리스트 [(x1,y1), (x2,y2), ...]
    """
    if len(lines) == 0:
        return [], [], []

    # 세로선, 가로선, 대각선 분리
    vertical_lines = []
    horizontal_lines = []
    diagonal_lines = []

    for line in lines:
        direction = classify_line_direction(line)
        if direction == "vertical":
            vertical_lines.append(line)
        elif direction == "horizontal":
            horizontal_lines.append(line)
        else:  # diagonal
            diagonal_lines.append(line)

    # 각각 병합
    # 세로선과 가로선은 각도 체크 없이 병합
    vertical_polylines = merge_connected_lines(vertical_lines, distance_threshold, check_angle=False)
    horizontal_polylines = merge_connected_lines(horizontal_lines, distance_threshold, check_angle=False)
    # 대각선은 각도 체크를 활성화하여 같은 방향의 선만 병합
    diagonal_polylines = merge_connected_lines(diagonal_lines, distance_threshold, check_angle=True, angle_threshold=30.0)

    return vertical_polylines, horizontal_polylines, diagonal_polylines


def merge_connected_lines(
    lines: list,
    distance_threshold: float = ENDPOINT_DISTANCE_THRESHOLD,
    check_angle: bool = False,
    angle_threshold: float = 30.0
) -> list[list[tuple[float, float]]]:
    """
    끝점이 가까운 선분들을 하나의 폴리라인으로 병합한다.

    Args:
        lines: 검출된 선분들 [[x1, y1, x2, y2], ...]
        distance_threshold: 끝점 사이의 최대 연결 거리 (픽셀)
        check_angle: True이면 각도가 유사한 선분만 병합 (대각선용)
        angle_threshold: 병합 허용 최대 각도 차이 (도)

    Returns:
        List of polylines, 각 폴리라인은 점들의 리스트 [(x1,y1), (x2,y2), ...]
    """
    if len(lines) == 0:
        return []

    # 각 선분을 (시작점, 끝점) 튜플로 변환
    segments = []
    segment_angles = []  # 각 선분의 각도 저장
    for line in lines:
        x1, y1, x2, y2 = line[:4]
        segments.append([(x1, y1), (x2, y2)])

        # 각도 계산 (필요한 경우)
        if check_angle:
            angle = calculate_line_angle(line)
            segment_angles.append(angle)
        else:
            segment_angles.append(0.0)  # 사용하지 않음

    # Union-Find 자료구조로 연결된 선분 그룹화
    parent = list(range(len(segments)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    # 모든 선분 쌍을 검사하여 끝점이 가까우면 연결
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg_i = segments[i]
            seg_j = segments[j]

            # 각도 체크 (대각선인 경우)
            if check_angle:
                angle_i = segment_angles[i]
                angle_j = segment_angles[j]
                angle_diff = abs(math.degrees(angle_i - angle_j))

                # 각도 차이를 [0, 90] 범위로 정규화
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                # 각도 차이가 임계값보다 크면 병합하지 않음
                if angle_diff > angle_threshold:
                    continue

            # 4가지 끝점 조합 확인
            endpoints_i = [seg_i[0], seg_i[1]]
            endpoints_j = [seg_j[0], seg_j[1]]

            connected = False
            for pt_i in endpoints_i:
                for pt_j in endpoints_j:
                    dist = math.sqrt((pt_i[0] - pt_j[0])**2 + (pt_i[1] - pt_j[1])**2)
                    if dist < distance_threshold:
                        union(i, j)
                        connected = True
                        break
                if connected:
                    break

    # 그룹별로 선분 수집
    groups = {}
    for i in range(len(segments)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # 각 그룹을 폴리라인으로 변환
    polylines = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            # 단일 선분은 그대로 폴리라인으로
            idx = group_indices[0]
            polylines.append([segments[idx][0], segments[idx][1]])
        else:
            # 여러 선분을 순서대로 연결
            polyline = _build_polyline([segments[i] for i in group_indices], distance_threshold)
            polylines.append(polyline)

    return polylines


def _build_polyline(segments: list, distance_threshold: float) -> list[tuple[float, float]]:
    """
    연결된 선분들을 순서대로 정렬하여 폴리라인 구성
    """
    if len(segments) == 0:
        return []

    if len(segments) == 1:
        return [segments[0][0], segments[0][1]]

    # 시작 선분 선택 (임의로 첫 번째)
    polyline = list(segments[0])  # [(x1,y1), (x2,y2)]
    used = {0}

    # 나머지 선분들을 순서대로 연결
    while len(used) < len(segments):
        last_point = polyline[-1]

        # 가장 가까운 미사용 선분 찾기
        best_idx = None
        best_dist = float('inf')
        best_reversed = False

        for i, seg in enumerate(segments):
            if i in used:
                continue

            # seg의 시작점이 last_point에 가까운지
            dist_start = math.sqrt((seg[0][0] - last_point[0])**2 + (seg[0][1] - last_point[1])**2)
            if dist_start < best_dist:
                best_dist = dist_start
                best_idx = i
                best_reversed = False

            # seg의 끝점이 last_point에 가까운지
            dist_end = math.sqrt((seg[1][0] - last_point[0])**2 + (seg[1][1] - last_point[1])**2)
            if dist_end < best_dist:
                best_dist = dist_end
                best_idx = i
                best_reversed = True

        # 연결할 선분이 없으면 종료
        if best_idx is None or best_dist > distance_threshold:
            break

        # 선분 추가
        if best_reversed:
            # 역순으로 추가 (끝점이 last_point에 가까움)
            polyline.append(segments[best_idx][0])
        else:
            # 정순으로 추가 (시작점이 last_point에 가까움)
            polyline.append(segments[best_idx][1])

        used.add(best_idx)

    return polyline


class SmoothedLineDetector:
    """
    Line detector with temporal smoothing.
    Detects lines and applies exponential moving average smoothing to stabilize results.
    """

    def __init__(self, smoothing_alpha: float = SMOOTHING_ALPHA, min_confidence: int = MIN_DETECTION_CONFIDENCE):
        """
        Initialize smoothed line detector.

        Args:
            smoothing_alpha: EMA smoothing factor (0=full smoothing, 1=no smoothing)
            min_confidence: Minimum number of consecutive detections before applying smoothing
        """
        self.smoothing_alpha = smoothing_alpha
        self.min_confidence = min_confidence

        # Smoothing state for vertical, horizontal, and diagonal polylines
        self.smoothed_vertical = []  # List of smoothed vertical polylines
        self.smoothed_horizontal = []  # List of smoothed horizontal polylines
        self.smoothed_diagonal = []  # List of smoothed diagonal polylines
        self.detection_count = 0  # Number of consecutive successful detections

    def _smooth_polylines(
        self, new_polylines: list[list[tuple[float, float]]], prev_polylines: list[list[tuple[float, float]]]
    ) -> list[list[tuple[float, float]]]:
        """
        Apply EMA smoothing to polylines.

        Args:
            new_polylines: Newly detected polylines
            prev_polylines: Previously smoothed polylines

        Returns:
            Smoothed polylines
        """
        if not prev_polylines or len(new_polylines) != len(prev_polylines):
            # If no previous or different number of polylines, return new ones directly
            return new_polylines

        smoothed = []
        alpha = self.smoothing_alpha

        for new_poly, old_poly in zip(new_polylines, prev_polylines):
            # Check if polylines have same number of points
            if len(new_poly) != len(old_poly):
                # Different structure, use new polyline
                smoothed.append(new_poly)
                continue

            # Apply EMA to each point
            smoothed_poly = []
            for new_pt, old_pt in zip(new_poly, old_poly):
                smoothed_x = alpha * new_pt[0] + (1 - alpha) * old_pt[0]
                smoothed_y = alpha * new_pt[1] + (1 - alpha) * old_pt[1]
                smoothed_poly.append((smoothed_x, smoothed_y))

            smoothed.append(smoothed_poly)

        return smoothed

    def detect_and_smooth(
        self, image: np.ndarray
    ) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
        """
        Detect lines in image and apply temporal smoothing.

        Args:
            image: Input BGR image

        Returns:
            (smoothed_vertical_polylines, smoothed_horizontal_polylines, smoothed_diagonal_polylines)
        """
        height, width = image.shape[:2]

        # Detect lines
        lines, _ = detect_lines_in_roi(image)

        # Get parallel pairs and centerlines
        parallel_pairs = select_all_parallel_line_pairs(lines, width, height)

        center_lines = []
        if parallel_pairs:
            for line1, line2 in parallel_pairs:
                center_line = calculate_center_line(line1, line2)
                center_lines.append(center_line)

        # Merge by direction (vertical, horizontal, diagonal)
        vertical_polylines = []
        horizontal_polylines = []
        diagonal_polylines = []
        if center_lines:
            vertical_polylines, horizontal_polylines, diagonal_polylines = merge_connected_lines_by_direction(center_lines)

        # Apply smoothing if we have enough confidence
        if vertical_polylines or horizontal_polylines or diagonal_polylines:
            self.detection_count += 1
        else:
            # No lines detected, reset
            self.detection_count = 0
            self.smoothed_vertical = []
            self.smoothed_horizontal = []
            self.smoothed_diagonal = []
            return [], [], []

        # Apply smoothing if confidence is high enough
        if self.detection_count >= self.min_confidence:
            smoothed_vertical = self._smooth_polylines(vertical_polylines, self.smoothed_vertical)
            smoothed_horizontal = self._smooth_polylines(horizontal_polylines, self.smoothed_horizontal)
            smoothed_diagonal = self._smooth_polylines(diagonal_polylines, self.smoothed_diagonal)
        else:
            # Still warming up, use raw detections
            smoothed_vertical = vertical_polylines
            smoothed_horizontal = horizontal_polylines
            smoothed_diagonal = diagonal_polylines

        # Update smoothing state
        self.smoothed_vertical = smoothed_vertical
        self.smoothed_horizontal = smoothed_horizontal
        self.smoothed_diagonal = smoothed_diagonal

        return smoothed_vertical, smoothed_horizontal, smoothed_diagonal

    def reset(self):
        """Reset smoothing state."""
        self.smoothed_vertical = []
        self.smoothed_horizontal = []
        self.smoothed_diagonal = []
        self.detection_count = 0
