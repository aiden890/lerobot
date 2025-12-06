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
Line following for LeKiwi using vertical and horizontal line alignment.

The robot detects vertical and horizontal lines, then aligns with the vertical line.

Usage:
    python -m lerobot.robots.lekiwi.follow_line
"""

import logging
import math
import time
from enum import Enum

import cv2

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.robots.lekiwi import line_detection

# ============================================================================
# 파라미터 설정 (Parameter Configuration)
# ============================================================================

# PD 컨트롤러 파라미터 (PD Controller Parameters)
LOOKAHEAD_DISTANCE = 100.0   # Lookahead distance (픽셀) - 목표 지점 설정용
KX = 0.001                   # X 방향 비례 게인 (전진/후진)
KY = 0.001                   # Y 방향 비례 게인 (좌우)
KTH = 0.03                    # 회전 비례 게인 (theta)
MAX_VX = 0.1                # 최대 전진 속도 (m/s)
MAX_VY = 0.02                # 최대 좌우 속도 (m/s)
MAX_OMEGA = 20.0             # 최대 회전 속도 (deg/s)

# 테스트 모드 설정 (Test Mode Configuration)
# "all": 모든 제어 활성화 (전진 + 좌우 + 회전)
# "lateral_only": 좌우 편차 제어만 (y.vel만 활성화, x.vel=0, theta.vel=0)
# "heading_only": 각도 편차 제어만 (theta.vel만 활성화, x.vel=0, y.vel=0)
# "forward_only": 전진만 (x.vel만 활성화, y.vel=0, theta.vel=0)
TEST_MODE = "all"

# 가로선 감지 시 동작 파라미터 (Horizontal Line Action Parameters)
FORWARD_SPEED = 0.1                       # 전진 속도 (m/s)
FORWARD_DISTANCE_ON_HORIZONTAL = 0.4      # 가로선 감지 시 전진할 거리 (m) - 50cm
TURN_ANGLE = 60.0                         # 회전할 각도 (도)
TURN_SPEED = 40.0                         # 회전 속도 (deg/s)

# 거리 조절 파라미터 (Distance Modulation Parameters)
# 공식: 실제 이동 거리 = 기본 거리 - a * 상하 거리 (normalized, 0~1)
DISTANCE_REDUCTION_COEFF = 0.14           # 거리 감소 계수 a (m) - 14cm
                                           # ROI 상단(0): 감소 없음 (원래 거리)
                                           # ROI 하단(1): 14cm 감소
                                           # ROI 중간(0.5): 7cm 감소
                                           # 예: 기본 0.4m → ROI 상단: 0.4m, 중간: 0.33m, 하단: 0.26m

# 대각선 감지 시 동작 파라미터 (Diagonal Line Action Parameters)
# /\ 방향 (도착 지점)
FORWARD_DISTANCE_ON_DIAGONAL = 0.55       # 대각선 감지 시 전진할 거리 (m)
LATERAL_DISTANCE_ON_DIAGONAL = 0.12       # 대각선 감지 시 수평이동 거리 (m) - 20cm (오른쪽)
LATERAL_SPEED_ON_DIAGONAL = 0.07          # 대각선 감지 시 수평이동 속도 (m/s)

# \/ 방향 (초기 지점 복귀)
FORWARD_DISTANCE_ON_INITIAL = 0.3         # 초기 지점 복귀 시 전진할 거리 (m) - 30cm
TURN_ANGLE_ON_INITIAL = 130.0             # 초기 지점 복귀 시 회전할 각도 (도)
TURN_SPEED_ON_INITIAL = 40.0              # 초기 지점 복귀 시 회전 속도 (deg/s)

# ARRIVED 상태에서 재개 시 (Resume from ARRIVED state)
LATERAL_DISTANCE_ON_RESUME = LATERAL_DISTANCE_ON_DIAGONAL  # 도착 지점에서 재개 시 왼쪽 이동 거리 (m)
LATERAL_SPEED_ON_RESUME = LATERAL_SPEED_ON_DIAGONAL        # 도착 지점에서 재개 시 왼쪽 이동 속도 (m/s)
TURN_ANGLE_ON_RESUME = 130.0                               # 도착 지점에서 재개 시 회전할 각도 (도)
TURN_SPEED_ON_RESUME = 40.0                                # 도착 지점에서 재개 시 회전 속도 (deg/s)

# 로봇 연결 설정
ROBOT_IP = "192.168.20.199"  # LeKiwi 로봇 IP 주소
CONTROL_HZ = 30.0            # 제어 루프 주파수 (Hz)

# ============================================================================
# 라인 detection 파라미터는 line_detection.py에서 설정
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class FollowState(Enum):
    """로봇의 상태를 정의하는 Enum"""
    INITIAL_TURN = "initial_turn"               # 초기 회전 (line following 시작 전 130도 회전)
    FOLLOW_VERTICAL = "follow_vertical"         # 세로선 추종
    HORIZONTAL_ACTION = "horizontal_action"     # 가로선 액션 (직진 + 회전)
    DIAGONAL_ACTION = "diagonal_action"         # 대각선 액션 (직진 + 회전) - /\ 방향 도착
    INITIAL_ACTION = "initial_action"           # 초기 지점 복귀 액션 (직진 + 회전) - \/ 방향
    ARRIVED = "arrived"                         # 도착
    INITIAL = "initial"                         # 초기 지점 복귀 완료


class VerticalLineAligner:
    """Aligns LeKiwi robot with a detected vertical line."""

    def __init__(self, robot: LeKiwiClient, control_hz: float = CONTROL_HZ):
        self.robot = robot
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

        # PD Controller parameters
        self.lookahead_distance = LOOKAHEAD_DISTANCE
        self.kx = KX
        self.ky = KY
        self.kth = KTH
        self.max_vx = MAX_VX
        self.max_vy = MAX_VY
        self.max_omega = MAX_OMEGA
        self.forward_speed = FORWARD_SPEED

        # Line detector with smoothing
        self.line_detector = line_detection.SmoothedLineDetector()

        # State machine variables
        self.state = FollowState.FOLLOW_VERTICAL
        self.action_phase = "forward"     # 액션 상태의 하위 단계: "forward" 또는 "turn"
        self.distance_traveled = 0.0      # 이동한 거리 (m)
        self.angle_turned = 0.0           # 회전한 각도 (deg)
        self.turn_direction = 1.0         # 회전 방향 (1.0 = 반시계, -1.0 = 시계)
        self.horizontal_detected = False  # 가로선 감지 여부
        self.diagonal_detected = False    # 대각선 쌍 감지 여부
        self.target_forward_distance = 0.0  # 동적으로 계산된 목표 전진 거리 (m)
        self.lateral_distance_traveled = 0.0  # 수평 이동한 거리 (m)

    def select_best_vertical_line(self, vertical_polylines: list) -> list | None:
        """
        가장 적합한 세로선을 선택한다.
        현재는 가장 긴 세로선을 선택.

        Returns:
            Best polyline as [(x1,y1), (x2,y2), ...] or None
        """
        if not vertical_polylines:
            return None

        best_polyline = None
        best_length = 0.0

        for polyline in vertical_polylines:
            # Calculate total length of polyline
            total_length = 0.0
            for i in range(len(polyline) - 1):
                pt1 = polyline[i]
                pt2 = polyline[i + 1]
                segment_length = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                total_length += segment_length

            if total_length > best_length:
                best_length = total_length
                best_polyline = polyline

        return best_polyline

    def find_closest_point_on_polyline(
        self, polyline: list, robot_x: float, robot_y: float
    ) -> tuple[tuple[float, float], int, float]:
        """
        Polyline 위에서 로봇과 가장 가까운 점을 찾는다.

        Args:
            polyline: 점들의 리스트 [(x1,y1), (x2,y2), ...]
            robot_x, robot_y: 로봇 위치

        Returns:
            (closest_point, segment_index, distance_along_segment)
            - closest_point: 가장 가까운 점 (x, y)
            - segment_index: 해당 점이 속한 선분의 인덱스
            - distance_along_segment: 선분 시작점부터의 거리 (0.0 ~ segment_length)
        """
        min_dist = float('inf')
        closest_point = polyline[0]
        closest_segment_idx = 0
        distance_along = 0.0

        # Check each segment of the polyline
        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i + 1]

            # Vector from p1 to p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            segment_length = math.sqrt(dx * dx + dy * dy)

            if segment_length < 1e-6:
                # Degenerate segment, use p1
                dist = math.sqrt((robot_x - p1[0])**2 + (robot_y - p1[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = p1
                    closest_segment_idx = i
                    distance_along = 0.0
                continue

            # Project robot position onto the line segment
            # Parameter t: 0 = p1, 1 = p2
            t = ((robot_x - p1[0]) * dx + (robot_y - p1[1]) * dy) / (segment_length * segment_length)
            t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

            # Closest point on this segment
            proj_x = p1[0] + t * dx
            proj_y = p1[1] + t * dy

            # Distance to this projected point
            dist = math.sqrt((robot_x - proj_x)**2 + (robot_y - proj_y)**2)

            if dist < min_dist:
                min_dist = dist
                closest_point = (proj_x, proj_y)
                closest_segment_idx = i
                distance_along = t * segment_length

        return closest_point, closest_segment_idx, distance_along

    def find_lookahead_point(
        self, polyline: list, start_segment_idx: int, distance_along_segment: float, lookahead_dist: float
    ) -> tuple[float, float] | None:
        """
        시작점에서 lookahead distance만큼 떨어진 점을 polyline을 따라 찾는다.

        Args:
            polyline: 점들의 리스트 [(x1,y1), (x2,y2), ...]
            start_segment_idx: 시작 선분 인덱스
            distance_along_segment: 선분 시작점부터의 거리
            lookahead_dist: 찾고자 하는 거리

        Returns:
            Lookahead point (x, y) or None if not found
        """
        remaining_dist = lookahead_dist

        # Start from the current segment
        for i in range(start_segment_idx, len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i + 1]

            # For the first segment, account for distance already traveled
            if i == start_segment_idx:
                # Start from the point at distance_along_segment
                segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                remaining_in_segment = segment_length - distance_along_segment

                if remaining_in_segment >= remaining_dist:
                    # Lookahead point is in this segment
                    total_dist = distance_along_segment + remaining_dist
                    t = total_dist / segment_length if segment_length > 1e-6 else 0.0
                    lookahead_x = p1[0] + t * (p2[0] - p1[0])
                    lookahead_y = p1[1] + t * (p2[1] - p1[1])
                    return (lookahead_x, lookahead_y)
                else:
                    remaining_dist -= remaining_in_segment
            else:
                # Subsequent segments
                segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

                if segment_length >= remaining_dist:
                    # Lookahead point is in this segment
                    t = remaining_dist / segment_length if segment_length > 1e-6 else 0.0
                    lookahead_x = p1[0] + t * (p2[0] - p1[0])
                    lookahead_y = p1[1] + t * (p2[1] - p1[1])
                    return (lookahead_x, lookahead_y)
                else:
                    remaining_dist -= segment_length

        # If we reach here, lookahead distance exceeds the polyline length
        # Return the last point of the polyline
        return polyline[-1]

    def compute_pd_control(
        self, vertical_polylines: list, image_width: int, image_height: int
    ) -> dict | None:
        """
        PD 컨트롤러를 사용하여 세로선을 따라가는 제어 명령을 계산한다.

        전방향(omnidirectional) 로봇을 위한 PD 컨트롤러:
        1. 현재 로봇 위치와 목표 위치의 오차 계산
        2. 오차를 로봇 body 좌표계로 변환
        3. 비례 제어로 x.vel, y.vel, theta.vel 계산

        Args:
            vertical_polylines: 검출된 세로선 polylines
            image_width: 이미지 너비
            image_height: 이미지 높이

        Returns:
            Tuple of (action, lateral_error, lookahead_point, closest_point, best_line) or None if no line
        """
        # Select best vertical line
        best_line = self.select_best_vertical_line(vertical_polylines)
        if best_line is None:
            return None

        # Current robot pose in image coordinates
        # 로봇은 이미지 하단 중앙에 위치하며, 위쪽(-y 방향)을 바라봄
        robot_x = image_width / 2
        robot_y = image_height
        robot_theta = 0.0  # 위쪽을 바라봄 (in atan2(dx, -dy) convention, 0 = upward)

        # Find closest point on the polyline
        closest_point, segment_idx, dist_along = self.find_closest_point_on_polyline(
            best_line, robot_x, robot_y
        )

        # Find lookahead point as target
        lookahead_point = self.find_lookahead_point(
            best_line, segment_idx, dist_along, self.lookahead_distance
        )

        if lookahead_point is None:
            return None

        # Target pose
        target_x = lookahead_point[0]
        target_y = lookahead_point[1]

        # Target heading: align with the line direction
        # 세로선의 실제 방향을 계산
        line_start = best_line[0]
        line_end = best_line[-1]
        dx_line = line_end[0] - line_start[0]
        dy_line = line_end[1] - line_start[1]

        # Calculate line orientation (angle from upward direction)
        # atan2(dx, -dy) gives angle from vertical axis (upward = -y direction)
        target_theta = math.atan2(dx_line, -dy_line)

        # Position error in world (image) frame
        dx = target_x - robot_x
        dy = target_y - robot_y

        # Transform error to robot body frame
        # Body frame: x = forward, y = left
        # Note: robot_theta is in atan2(dx, -dy) convention where 0 = upward
        # Convert to standard convention where 0 = rightward by adding -π/2
        robot_theta_standard = robot_theta - math.pi / 2
        cos_theta = math.cos(robot_theta_standard)
        sin_theta = math.sin(robot_theta_standard)

        # Standard rotation matrix for body frame transformation
        # Forward direction: (cos(θ), sin(θ))
        # Left direction (90° CCW): (-sin(θ), cos(θ))
        ex_b = cos_theta * dx + sin_theta * dy   # Forward error
        ey_b = -sin_theta * dx + cos_theta * dy  # Lateral error

        # Heading error
        e_theta = target_theta - robot_theta
        # Wrap to [-pi, pi]
        e_theta = math.atan2(math.sin(e_theta), math.cos(e_theta))

        # PD control (proportional control)
        vx = self.kx * ex_b
        vy = -self.ky * ey_b
        omega_rad = self.kth * e_theta
        omega_deg = math.degrees(omega_rad)

        # Clamp velocities
        vx = max(min(vx, self.max_vx), -self.max_vx)
        vy = max(min(vy, self.max_vy), -self.max_vy)
        omega_deg = max(min(omega_deg, self.max_omega), -self.max_omega)

        # Test mode: disable certain velocities for isolated testing
        if TEST_MODE == "lateral_only":
            # 좌우 편차 제어만 테스트
            vx = 0.0
            omega_deg = 0.0
        elif TEST_MODE == "heading_only":
            # 각도 편차 제어만 테스트
            vx = 0.0
            vy = 0.0
        elif TEST_MODE == "forward_only":
            # 전진만 테스트
            vy = 0.0
            omega_deg = 0.0
        # TEST_MODE == "all"이면 모든 제어 활성화 (변경 없음)

        # Get current observation for arm positions
        observation = self.robot.get_observation()

        action = {
            "x.vel": vx,
            "y.vel": vy,
            "theta.vel": omega_deg,
            "arm_shoulder_pan.pos": observation.get("arm_shoulder_pan.pos", 0.0),
            "arm_shoulder_lift.pos": observation.get("arm_shoulder_lift.pos", 0.0),
            "arm_elbow_flex.pos": observation.get("arm_elbow_flex.pos", 0.0),
            "arm_wrist_flex.pos": observation.get("arm_wrist_flex.pos", 0.0),
            "arm_wrist_roll.pos": observation.get("arm_wrist_roll.pos", 0.0),
            "arm_gripper.pos": observation.get("arm_gripper.pos", 0.0),
        }

        # Calculate lateral error for display (in pixels)
        lateral_error_px = closest_point[0] - robot_x

        logger.debug(
            f"PD Control: ex_b={ex_b:.3f}m, ey_b={ey_b:.3f}m, e_theta={math.degrees(e_theta):.1f}°, "
            f"vx={vx:.3f}, vy={vy:.3f}, omega={omega_deg:.1f}"
        )

        return action, lateral_error_px, lookahead_point, closest_point, best_line

    def determine_turn_direction(self, horizontal_polylines: list, image_width: int) -> float:
        """
        가로선의 위치를 기반으로 회전 방향을 결정한다.

        Args:
            horizontal_polylines: 감지된 가로선들
            image_width: 이미지 너비

        Returns:
            1.0 (반시계방향, 왼쪽) 또는 -1.0 (시계방향, 오른쪽)
        """
        if not horizontal_polylines:
            return 1.0  # Default: 반시계방향

        # 첫 번째 가로선의 중심점 계산
        best_line = horizontal_polylines[0]
        center_x = sum(pt[0] for pt in best_line) / len(best_line)

        # 이미지 중심을 기준으로 왼쪽/오른쪽 판단
        image_center = image_width / 2

        if center_x < image_center:
            # 가로선이 왼쪽에 있음 → 왼쪽으로 회전 (반시계방향)
            return 1.0
        else:
            # 가로선이 오른쪽에 있음 → 오른쪽으로 회전 (시계방향)
            return -1.0

    def calculate_adjusted_forward_distance(
        self, polylines: list, base_distance: float, image_height: int
    ) -> float:
        """
        선의 y 위치에 따라 조정된 전진 거리를 계산한다.

        공식: 조정된 거리 = base_distance - a * normalized_y_position

        Args:
            polylines: 감지된 선들 (가로선 또는 대각선)
            base_distance: 기본 전진 거리 (m)
            image_height: 이미지 높이 (픽셀)

        Returns:
            조정된 전진 거리 (m), 최소값은 base_distance * 0.1
        """
        if not polylines:
            return base_distance

        # ROI 범위 (line_detection.py의 파라미터 사용)
        from lerobot.robots.lekiwi import line_detection
        roi_y_top = line_detection.ROI_Y_TOP * image_height
        roi_y_bottom = line_detection.ROI_Y_BOTTOM * image_height
        roi_height = roi_y_bottom - roi_y_top

        if roi_height < 1:
            return base_distance

        # 모든 선의 평균 y 좌표 계산
        total_y = 0
        point_count = 0
        for polyline in polylines:
            for point in polyline:
                total_y += point[1]
                point_count += 1

        if point_count == 0:
            return base_distance

        avg_y = total_y / point_count

        # ROI 내에서의 정규화된 위치 (0: 상단, 1: 하단)
        normalized_y = (avg_y - roi_y_top) / roi_height
        normalized_y = max(0.0, min(1.0, normalized_y))  # 0~1로 클램핑

        # 조정된 거리 = 기본 거리 - a * 정규화된 y 위치
        adjusted_distance = base_distance - DISTANCE_REDUCTION_COEFF * normalized_y

        # 최소값 보장 (안전을 위해 최소 5cm)
        min_distance = 0.05  # 5cm
        adjusted_distance = max(min_distance, adjusted_distance)

        return adjusted_distance

    def detect_diagonal_pair(self, diagonal_polylines: list) -> tuple[bool, str]:
        """
        상대적 위치를 기준으로 양쪽에 대각선이 하나씩 있는지 확인하고 방향을 판단한다.

        Args:
            diagonal_polylines: 감지된 대각선들

        Returns:
            (pair_detected, direction) 튜플
            - pair_detected: True if 대각선 쌍이 양쪽에 감지됨
            - direction: "up" (/\) 또는 "down" (\/) 또는 "unknown"
        """
        if len(diagonal_polylines) < 2:
            return False, "unknown"

        # 각 대각선의 중심 x 좌표 계산 및 정렬
        polylines_with_center = []
        for polyline in diagonal_polylines:
            center_x = sum(pt[0] for pt in polyline) / len(polyline)
            polylines_with_center.append((center_x, polyline))

        # x 좌표로 정렬
        polylines_with_center.sort(key=lambda x: x[0])

        # 상대적 위치로 왼쪽/오른쪽 분할
        mid_index = len(polylines_with_center) // 2
        left_lines = [pl for _, pl in polylines_with_center[:mid_index]]
        right_lines = [pl for _, pl in polylines_with_center[mid_index:]]

        # 양쪽에 최소 하나씩 있어야 함
        if len(left_lines) < 1 or len(right_lines) < 1:
            return False, "unknown"

        # 방향 판단: 대각선들의 기울기로 판단
        # 이미지 좌표계에서 (y는 아래가 양수):
        # /\ 방향 (up/arrival): 왼쪽은 / (slope < 0), 오른쪽은 \ (slope > 0)
        # \/ 방향 (down/initial): 왼쪽은 \ (slope > 0), 오른쪽은 / (slope < 0)

        def get_slope(polyline):
            """polyline의 평균 기울기 계산 (dy/dx)"""
            if len(polyline) < 2:
                return 0

            # 모든 점의 평균 위치로 시작점과 끝점 계산
            x_coords = [pt[0] for pt in polyline]
            y_coords = [pt[1] for pt in polyline]

            dx = max(x_coords) - min(x_coords)
            dy = max(y_coords) - min(y_coords)

            if dx < 1e-6:  # 거의 수직선
                return 0

            # 왼쪽에서 오른쪽으로 갈 때의 기울기
            # polyline의 좌우 끝점 찾기
            leftmost_idx = x_coords.index(min(x_coords))
            rightmost_idx = x_coords.index(max(x_coords))

            dx = x_coords[rightmost_idx] - x_coords[leftmost_idx]
            dy = y_coords[rightmost_idx] - y_coords[leftmost_idx]

            return dy / dx if dx != 0 else 0

        # 왼쪽과 오른쪽 대각선의 평균 기울기 계산
        left_slopes = [get_slope(line) for line in left_lines]
        right_slopes = [get_slope(line) for line in right_lines]

        avg_left_slope = sum(left_slopes) / len(left_slopes) if left_slopes else 0
        avg_right_slope = sum(right_slopes) / len(right_slopes) if right_slopes else 0

        # 방향 결정
        # /\ 패턴: 왼쪽 < 0 (올라감), 오른쪽 > 0 (내려감)
        # \/ 패턴: 왼쪽 > 0 (내려감), 오른쪽 < 0 (올라감)
        if avg_left_slope < -0.3 and avg_right_slope > 0.3:
            direction = "up"  # /\ 패턴 (도착)
        elif avg_left_slope > 0.3 and avg_right_slope < -0.3:
            direction = "down"  # \/ 패턴 (초기 위치)
        else:
            direction = "unknown"

        return True, direction

    def _make_base_action(self, x_vel: float = 0.0, y_vel: float = 0.0, theta_vel: float = 0.0) -> dict:
        """Base velocities와 현재 arm positions로 action dict 생성."""
        obs = self.robot.get_observation()
        return {
            "x.vel": x_vel,
            "y.vel": y_vel,
            "theta.vel": theta_vel,
            "arm_shoulder_pan.pos": obs.get("arm_shoulder_pan.pos", 0.0),
            "arm_shoulder_lift.pos": obs.get("arm_shoulder_lift.pos", 0.0),
            "arm_elbow_flex.pos": obs.get("arm_elbow_flex.pos", 0.0),
            "arm_wrist_flex.pos": obs.get("arm_wrist_flex.pos", 0.0),
            "arm_wrist_roll.pos": obs.get("arm_wrist_roll.pos", 0.0),
            "arm_gripper.pos": obs.get("arm_gripper.pos", 0.0),
        }

    def compute_forward_control(self) -> dict:
        """직진 제어 명령을 계산한다."""
        return self._make_base_action(x_vel=FORWARD_SPEED)

    def compute_turn_control(self, turn_speed: float) -> dict:
        """회전 제어 명령을 계산한다."""
        return self._make_base_action(theta_vel=self.turn_direction * turn_speed)

    def compute_lateral_control(self, lateral_speed: float, direction: float = -1.0) -> dict:
        """수평 이동 제어 명령을 계산한다."""
        return self._make_base_action(y_vel=direction * lateral_speed)

    def update_state(
        self,
        horizontal_polylines: list,
        diagonal_polylines: list,
        image_width: int,
        image_height: int,
        dt: float
    ) -> None:
        """
        상태 머신을 업데이트한다.

        Args:
            horizontal_polylines: 감지된 가로선들
            diagonal_polylines: 감지된 대각선들
            image_width: 이미지 너비
            image_height: 이미지 높이
            dt: 시간 간격 (초)
        """
        # INITIAL_TURN 상태: line following 시작 전 초기 회전 (130도)
        if self.state == FollowState.INITIAL_TURN:
            # 회전 단계: 각도 추적
            self.angle_turned += abs(TURN_SPEED_ON_RESUME * dt)

            if self.angle_turned >= TURN_ANGLE_ON_RESUME:
                # 목표 각도 도달 → FOLLOW_VERTICAL 상태로 전환
                self.state = FollowState.FOLLOW_VERTICAL
                self.action_phase = "forward"
                self.angle_turned = 0.0
                logger.info(f"Initial turn {self.angle_turned:.1f}° complete. Switching to FOLLOW_VERTICAL state.")

        # FOLLOW_VERTICAL 상태: 대각선 쌍 또는 가로선 감지 확인
        elif self.state == FollowState.FOLLOW_VERTICAL:
            # 대각선 쌍 감지 우선 확인
            diagonal_pair_detected, diagonal_direction = self.detect_diagonal_pair(diagonal_polylines)

            if diagonal_pair_detected and not self.diagonal_detected:
                # 대각선 쌍이 처음 감지됨
                self.diagonal_detected = True

                if diagonal_direction == "up":
                    # /\ 방향 → DIAGONAL_ACTION (도착)
                    # y 위치에 따라 전진 거리 조정 (상단: 30cm, 하단: 16cm)
                    self.target_forward_distance = self.calculate_adjusted_forward_distance(
                        diagonal_polylines, FORWARD_DISTANCE_ON_DIAGONAL, image_height
                    )
                    self.state = FollowState.DIAGONAL_ACTION
                    self.action_phase = "forward"
                    self.distance_traveled = 0.0
                    self.angle_turned = 0.0
                    self.turn_direction = -1.0  # 오른쪽으로 회전
                    logger.info(f"Diagonal pair (UP /\\) detected! Target distance: {self.target_forward_distance:.3f}m. Will move RIGHT {LATERAL_DISTANCE_ON_DIAGONAL:.2f}m")
                elif diagonal_direction == "down":
                    # \/ 방향 → INITIAL_ACTION (초기 지점 복귀)
                    # y 위치에 따라 전진 거리 조정 (상단: 30cm, 하단: 16cm)
                    self.target_forward_distance = self.calculate_adjusted_forward_distance(
                        diagonal_polylines, FORWARD_DISTANCE_ON_INITIAL, image_height
                    )
                    self.state = FollowState.INITIAL_ACTION
                    self.action_phase = "forward"
                    self.distance_traveled = 0.0
                    self.angle_turned = 0.0
                    self.turn_direction = -1.0  # 오른쪽으로 회전
                    logger.info(f"Diagonal pair (DOWN \\/) detected! Target distance: {self.target_forward_distance:.3f}m. Will turn RIGHT 145°")
                else:
                    # 방향을 알 수 없음 - 대각선 감지 플래그 리셋
                    self.diagonal_detected = False

            elif horizontal_polylines and not self.horizontal_detected and not diagonal_pair_detected:
                # 가로선이 처음 감지됨 (대각선 쌍이 없을 때만) → HORIZONTAL_ACTION 상태로 전환
                # y 위치에 따라 전진 거리 조정 (상단: 40cm, 하단: 26cm)
                self.target_forward_distance = self.calculate_adjusted_forward_distance(
                    horizontal_polylines, FORWARD_DISTANCE_ON_HORIZONTAL, image_height
                )
                self.horizontal_detected = True
                self.state = FollowState.HORIZONTAL_ACTION
                self.action_phase = "forward"
                self.distance_traveled = 0.0
                self.angle_turned = 0.0
                self.turn_direction = self.determine_turn_direction(horizontal_polylines, image_width)
                logger.info(f"Horizontal line detected! Target distance: {self.target_forward_distance:.3f}m. Turn direction: {'LEFT' if self.turn_direction > 0 else 'RIGHT'}")

        # HORIZONTAL_ACTION 상태: 직진 후 회전 (가로선 감지 시)
        elif self.state == FollowState.HORIZONTAL_ACTION:
            if self.action_phase == "forward":
                # 전진 단계: 거리 추적
                self.distance_traveled += FORWARD_SPEED * dt

                if self.distance_traveled >= self.target_forward_distance:
                    # 목표 거리 도달 → 회전 단계로 전환
                    self.action_phase = "turn"
                    self.angle_turned = 0.0
                    logger.info(f"Forward distance {self.distance_traveled:.3f}m reached (target: {self.target_forward_distance:.3f}m). Switching to turn phase.")

            elif self.action_phase == "turn":
                # 회전 단계: 각도 추적
                self.angle_turned += abs(TURN_SPEED * dt)

                if self.angle_turned >= TURN_ANGLE:
                    # 목표 각도 도달 → FOLLOW_VERTICAL 상태로 복귀
                    self.state = FollowState.FOLLOW_VERTICAL
                    self.action_phase = "forward"
                    self.horizontal_detected = False
                    logger.info(f"Turn angle {self.angle_turned:.1f}° reached. Returning to FOLLOW_VERTICAL state.")

        # DIAGONAL_ACTION 상태: 직진 후 수평 이동 (대각선 감지 시 - 도착)
        elif self.state == FollowState.DIAGONAL_ACTION:
            if self.action_phase == "forward":
                # 전진 단계: 거리 추적
                self.distance_traveled += FORWARD_SPEED * dt

                if self.distance_traveled >= self.target_forward_distance:
                    # 목표 거리 도달 → 수평 이동 단계로 전환
                    self.action_phase = "lateral"
                    self.lateral_distance_traveled = 0.0
                    logger.info(f"Forward distance {self.distance_traveled:.3f}m reached (target: {self.target_forward_distance:.3f}m). Switching to lateral phase.")

            elif self.action_phase == "lateral":
                # 수평 이동 단계: 거리 추적 (오른쪽)
                self.lateral_distance_traveled += LATERAL_SPEED_ON_DIAGONAL * dt

                if self.lateral_distance_traveled >= LATERAL_DISTANCE_ON_DIAGONAL:
                    # 목표 거리 도달 → ARRIVED 상태로 전환 (도착 지점)
                    self.state = FollowState.ARRIVED
                    self.action_phase = "forward"
                    self.diagonal_detected = False
                    logger.info(f"Lateral distance {self.lateral_distance_traveled:.3f}m reached. DESTINATION ARRIVED - Switching to ARRIVED state.")

        # INITIAL_ACTION 상태: 직진만 하고 정지 (초기 지점 복귀 - 회전 없음)
        elif self.state == FollowState.INITIAL_ACTION:
            if self.action_phase == "forward":
                # 전진 단계: 거리 추적
                self.distance_traveled += FORWARD_SPEED * dt

                if self.distance_traveled >= self.target_forward_distance:
                    # 목표 거리 도달 → 바로 INITIAL 상태로 전환 (회전 없이)
                    self.state = FollowState.INITIAL
                    self.action_phase = "forward"
                    self.diagonal_detected = False
                    logger.info(f"Forward distance {self.distance_traveled:.3f}m reached (target: {self.target_forward_distance:.3f}m). INITIAL POSITION REACHED - Switching to INITIAL state (no turn).")

        # ARRIVED 상태: 도착 지점에서 정지 또는 수평이동+회전 (재개)
        elif self.state == FollowState.ARRIVED:
            if self.action_phase == "lateral":
                # 수평 이동 단계: 거리 추적 (왼쪽)
                self.lateral_distance_traveled += LATERAL_SPEED_ON_RESUME * dt

                if self.lateral_distance_traveled >= LATERAL_DISTANCE_ON_RESUME:
                    # 목표 거리 도달 → 회전 단계로 전환
                    self.action_phase = "turn"
                    self.angle_turned = 0.0
                    logger.info(f"Resume lateral distance {self.lateral_distance_traveled:.3f}m reached. Switching to turn phase.")

            elif self.action_phase == "turn":
                # 회전 단계: 각도 추적
                self.angle_turned += abs(TURN_SPEED_ON_RESUME * dt)

                if self.angle_turned >= TURN_ANGLE_ON_RESUME:
                    # 목표 각도 도달 → FOLLOW_VERTICAL 상태로 복귀
                    self.state = FollowState.FOLLOW_VERTICAL
                    self.action_phase = "forward"
                    self.diagonal_detected = False
                    logger.info(f"Resume turn angle {self.angle_turned:.1f}° reached. Returning to FOLLOW_VERTICAL state.")

        # INITIAL 상태: 초기 지점에서 정지 또는 회전 (재개)
        elif self.state == FollowState.INITIAL:
            if self.action_phase == "turn":
                # 회전 단계: 각도 추적 (n 키를 눌러서 재개할 때 - 130° 회전)
                self.angle_turned += abs(TURN_SPEED_ON_INITIAL * dt)

                if self.angle_turned >= TURN_ANGLE_ON_INITIAL:
                    # 목표 각도 도달 → FOLLOW_VERTICAL 상태로 복귀
                    self.state = FollowState.FOLLOW_VERTICAL
                    self.action_phase = "forward"
                    self.diagonal_detected = False
                    logger.info(f"Resume turn angle {self.angle_turned:.1f}° reached. Returning to FOLLOW_VERTICAL state.")

    def stop(self) -> None:
        """Stop the robot."""
        self.robot.send_action(self._make_base_action())


def main():
    # Robot configuration
    robot_config = LeKiwiClientConfig(remote_ip=ROBOT_IP, id="lekiwi")
    robot = LeKiwiClient(robot_config)

    # Connect to robot
    logger.info("Connecting to LeKiwi...")
    robot.connect()
    logger.info("Connected!")

    # Create aligner
    aligner = VerticalLineAligner(robot, control_hz=CONTROL_HZ)

    # Control loop timing
    dt = 1.0 / CONTROL_HZ

    try:
        logger.info(f"Starting PD Controller line following at {CONTROL_HZ}Hz. Press 'q' to quit.")
        logger.info(f"Lookahead distance: {LOOKAHEAD_DISTANCE}px, KX: {KX}, KY: {KY}, KTH: {KTH}")
        logger.info(f"TEST MODE: {TEST_MODE}")

        # FPS and Timing display settings - update every 0.3 seconds for readability
        display_update_interval = 0.3  # 화면 업데이트 주기 (초)
        last_display_update_time = time.perf_counter()

        # FPS tracking
        displayed_fps = CONTROL_HZ  # 화면에 표시될 FPS
        fps_samples = []  # FPS 샘플들을 모아서 평균 계산

        # Timing tracking
        displayed_timing = {
            'obs': 0.0,
            'detect': 0.0,
            'control': 0.0,
            'action': 0.0,
            'visual': 0.0,
            'total': 0.0
        }
        timing_samples = {key: [] for key in displayed_timing.keys()}

        while True:
            start_time = time.perf_counter()

            # Timing measurements for performance analysis
            t_obs_start = time.perf_counter()

            # Get observation from robot
            observation = robot.get_observation()

            t_obs_end = time.perf_counter()

            # Get first camera image
            camera_name = list(robot.config.cameras.keys())[0]
            image = observation[camera_name]
            height, width = image.shape[:2]

            # Create visualization image
            vis_image = image.copy()

            t_detect_start = time.perf_counter()

            # Detect lines with smoothing (all processing done in line_detection)
            vertical_polylines, horizontal_polylines, diagonal_polylines = aligner.line_detector.detect_and_smooth(image)

            t_detect_end = time.perf_counter()

            # Draw ROI for reference
            _, roi_bounds = line_detection.detect_lines_in_roi(image)
            roi_x_start, roi_y_start, roi_x_end, roi_y_end = roi_bounds
            cv2.rectangle(vis_image, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

            # Draw vertical polylines in BLUE
            for polyline in vertical_polylines:
                for i in range(len(polyline) - 1):
                    pt1 = tuple(map(int, polyline[i]))
                    pt2 = tuple(map(int, polyline[i + 1]))
                    cv2.line(vis_image, pt1, pt2, (255, 0, 0), 5)

            # Draw horizontal polylines in ORANGE
            for polyline in horizontal_polylines:
                for i in range(len(polyline) - 1):
                    pt1 = tuple(map(int, polyline[i]))
                    pt2 = tuple(map(int, polyline[i + 1]))
                    cv2.line(vis_image, pt1, pt2, (0, 165, 255), 5)

            # Draw diagonal polylines in GREEN
            for polyline in diagonal_polylines:
                for i in range(len(polyline) - 1):
                    pt1 = tuple(map(int, polyline[i]))
                    pt2 = tuple(map(int, polyline[i + 1]))
                    cv2.line(vis_image, pt1, pt2, (0, 255, 0), 5)

            # Update state machine
            aligner.update_state(horizontal_polylines, diagonal_polylines, width, height, dt)

            t_control_start = time.perf_counter()

            # Execute control based on current state
            action = None
            state_info = {}

            if aligner.state == FollowState.INITIAL_TURN:
                # 초기 회전 (130도) - line following 시작 전
                action = aligner.compute_turn_control(TURN_SPEED_ON_RESUME)
                state_info = {
                    "type": "initial_turn",
                    "angle": aligner.angle_turned,
                    "target": TURN_ANGLE_ON_RESUME,
                    "direction": "LEFT" if aligner.turn_direction > 0 else "RIGHT",
                }

            elif aligner.state == FollowState.FOLLOW_VERTICAL:
                # PD 컨트롤러 세로선 추종
                if vertical_polylines:
                    result = aligner.compute_pd_control(vertical_polylines, width, height)
                    if result is not None:
                        action, lateral_error, lookahead_point, closest_point, best_line = result
                        state_info = {
                            "type": "pd_control",
                            "lateral_error": lateral_error,
                            "lookahead_point": lookahead_point,
                            "closest_point": closest_point,
                            "best_line": best_line
                        }

            elif aligner.state == FollowState.HORIZONTAL_ACTION:
                # 가로선 액션: 직진 또는 회전
                if aligner.action_phase == "forward":
                    action = aligner.compute_forward_control()
                    state_info = {
                        "type": "horizontal_forward",
                        "distance": aligner.distance_traveled,
                        "target": aligner.target_forward_distance,
                    }
                elif aligner.action_phase == "turn":
                    action = aligner.compute_turn_control(TURN_SPEED)
                    state_info = {
                        "type": "horizontal_turn",
                        "angle": aligner.angle_turned,
                        "target": TURN_ANGLE,
                        "direction": "LEFT" if aligner.turn_direction > 0 else "RIGHT",
                    }

            elif aligner.state == FollowState.DIAGONAL_ACTION:
                # 대각선 액션: 직진 또는 수평 이동 (도착)
                if aligner.action_phase == "forward":
                    action = aligner.compute_forward_control()
                    state_info = {
                        "type": "diagonal_forward",
                        "distance": aligner.distance_traveled,
                        "target": aligner.target_forward_distance,
                    }
                elif aligner.action_phase == "lateral":
                    action = aligner.compute_lateral_control(LATERAL_SPEED_ON_DIAGONAL, direction=-1.0)  # 오른쪽
                    state_info = {
                        "type": "diagonal_lateral",
                        "distance": aligner.lateral_distance_traveled,
                        "target": LATERAL_DISTANCE_ON_DIAGONAL,
                        "direction": "RIGHT",
                    }

            elif aligner.state == FollowState.INITIAL_ACTION:
                # 초기 지점 복귀 액션: 직진만 (회전 없음)
                action = aligner.compute_forward_control()
                state_info = {
                    "type": "initial_forward",
                    "distance": aligner.distance_traveled,
                    "target": aligner.target_forward_distance,
                }

            elif aligner.state == FollowState.ARRIVED:
                # 도착 - 로봇 정지 또는 수평이동+회전 (재개)
                if aligner.action_phase == "lateral":
                    # 수평 이동 중 (n 키를 눌러서 재개 - 왼쪽 이동)
                    action = aligner.compute_lateral_control(LATERAL_SPEED_ON_RESUME, direction=1.0)  # 왼쪽
                    state_info = {
                        "type": "arrived_lateral",
                        "distance": aligner.lateral_distance_traveled,
                        "target": LATERAL_DISTANCE_ON_RESUME,
                        "direction": "LEFT",
                    }
                elif aligner.action_phase == "turn":
                    # 회전 중
                    action = aligner.compute_turn_control(TURN_SPEED_ON_RESUME)
                    state_info = {
                        "type": "arrived_turn",
                        "angle": aligner.angle_turned,
                        "target": TURN_ANGLE_ON_RESUME,
                        "direction": "LEFT" if aligner.turn_direction > 0 else "RIGHT",
                    }
                else:
                    # 정지 상태
                    action = None  # stop() will be called
                    state_info = {
                        "type": "arrived"
                    }

            elif aligner.state == FollowState.INITIAL:
                # 초기 지점 복귀 완료 - 로봇 정지 또는 회전 (재개 시 130° 회전)
                if aligner.action_phase == "turn":
                    # 회전 중 (n 키를 눌러서 재개 - 130° 회전)
                    action = aligner.compute_turn_control(TURN_SPEED_ON_INITIAL)
                    state_info = {
                        "type": "initial_resume_turn",
                        "angle": aligner.angle_turned,
                        "target": TURN_ANGLE_ON_INITIAL,
                        "direction": "LEFT" if aligner.turn_direction > 0 else "RIGHT",
                    }
                else:
                    # 정지 상태
                    action = None  # stop() will be called
                    state_info = {
                        "type": "initial"
                    }

            t_control_end = time.perf_counter()

            t_action_start = time.perf_counter()

            # Send control command
            if action is not None:
                robot.send_action(action)
            else:
                aligner.stop()

            t_action_end = time.perf_counter()

            t_vis_start = time.perf_counter()

            # Draw robot position
            robot_x = width // 2
            robot_y = height
            cv2.circle(vis_image, (robot_x, robot_y), 10, (255, 255, 255), -1)
            cv2.circle(vis_image, (robot_x, robot_y), 12, (0, 0, 0), 2)

            # Draw state-specific visualization
            if state_info.get("type") == "pd_control":
                # Draw best line in GREEN (thick)
                best_line = state_info["best_line"]
                for i in range(len(best_line) - 1):
                    pt1 = tuple(map(int, best_line[i]))
                    pt2 = tuple(map(int, best_line[i + 1]))
                    cv2.line(vis_image, pt1, pt2, (0, 255, 0), 7)

                # Draw closest point on line (CYAN)
                closest_point = state_info["closest_point"]
                cv2.circle(vis_image, tuple(map(int, closest_point)), 8, (255, 255, 0), -1)
                cv2.line(vis_image, (robot_x, robot_y), tuple(map(int, closest_point)), (255, 255, 0), 2)

                # Draw lookahead point (MAGENTA)
                lookahead_point = state_info["lookahead_point"]
                cv2.circle(vis_image, tuple(map(int, lookahead_point)), 12, (255, 0, 255), -1)
                cv2.line(vis_image, (robot_x, robot_y), tuple(map(int, lookahead_point)), (255, 0, 255), 2)

                # Draw lookahead distance circle (faint)
                cv2.circle(vis_image, (robot_x, robot_y), int(LOOKAHEAD_DISTANCE), (128, 128, 128), 1)

            # Draw status text based on state
            if aligner.state == FollowState.INITIAL_TURN:
                if state_info.get("type") == "initial_turn":
                    status_text = f"INITIAL TURN - {state_info['direction']} ({state_info['angle']:.0f}/{state_info['target']:.0f}°)"
                    status_color = (255, 200, 0)  # Yellow-orange for initial turn
                else:
                    status_text = "INITIAL TURN"
                    status_color = (255, 200, 0)
            elif aligner.state == FollowState.FOLLOW_VERTICAL:
                status_text = "FOLLOW VERTICAL"
                status_color = (0, 255, 0)
                if state_info.get("type") == "pd_control":
                    lateral_error = state_info["lateral_error"]
                    if abs(lateral_error) > 40:
                        status_color = (0, 165, 255)
                    elif abs(lateral_error) > 20:
                        status_color = (0, 200, 255)
            elif aligner.state == FollowState.HORIZONTAL_ACTION:
                if state_info.get("type") == "horizontal_forward":
                    status_text = f"HORIZONTAL ACTION - FORWARD ({state_info['distance']:.2f}/{state_info['target']:.2f}m)"
                    status_color = (255, 165, 0)  # Orange
                elif state_info.get("type") == "horizontal_turn":
                    status_text = f"HORIZONTAL ACTION - TURN {state_info['direction']} ({state_info['angle']:.0f}/{state_info['target']:.0f}°)"
                    status_color = (255, 100, 0)  # Darker orange
                else:
                    status_text = "HORIZONTAL ACTION"
                    status_color = (255, 165, 0)
            elif aligner.state == FollowState.DIAGONAL_ACTION:
                if state_info.get("type") == "diagonal_forward":
                    status_text = f"DIAGONAL ACTION (/\\) - FORWARD ({state_info['distance']:.2f}/{state_info['target']:.2f}m)"
                    status_color = (255, 0, 255)  # Magenta
                elif state_info.get("type") == "diagonal_lateral":
                    status_text = f"DIAGONAL ACTION (/\\) - LATERAL {state_info['direction']} ({state_info['distance']:.2f}/{state_info['target']:.2f}m)"
                    status_color = (200, 0, 255)  # Darker magenta
                else:
                    status_text = "DIAGONAL ACTION (/\\)"
                    status_color = (255, 0, 255)
            elif aligner.state == FollowState.INITIAL_ACTION:
                if state_info.get("type") == "initial_forward":
                    status_text = f"INITIAL ACTION (\\/) - FORWARD ({state_info['distance']:.2f}/{state_info['target']:.2f}m)"
                    status_color = (0, 255, 255)  # Cyan
                else:
                    status_text = "INITIAL ACTION (\\/))"
                    status_color = (0, 255, 255)
            elif aligner.state == FollowState.ARRIVED:
                if state_info.get("type") == "arrived_lateral":
                    status_text = f"ARRIVED - RESUME LATERAL {state_info['direction']} ({state_info['distance']:.2f}/{state_info['target']:.2f}m)"
                    status_color = (0, 200, 0)  # Darker green for lateral
                elif state_info.get("type") == "arrived_turn":
                    status_text = f"ARRIVED - RESUME TURN {state_info['direction']} ({state_info['angle']:.0f}/{state_info['target']:.0f}°)"
                    status_color = (0, 180, 0)  # Even darker green for turning
                else:
                    status_text = "ARRIVED - DESTINATION REACHED (Press 'n' to continue)"
                    status_color = (0, 255, 0)  # Green for arrival
            elif aligner.state == FollowState.INITIAL:
                if state_info.get("type") == "initial_resume_turn":
                    status_text = f"INITIAL - RESUME TURN {state_info['direction']} ({state_info['angle']:.0f}/{state_info['target']:.0f}°)"
                    status_color = (200, 200, 0)  # Darker yellow for turning
                else:
                    status_text = "INITIAL - STARTING POINT (Press 'n' to continue)"
                    status_color = (255, 255, 0)  # Yellow for initial
            else:
                status_text = "UNKNOWN STATE"
                status_color = (0, 0, 255)

            cv2.putText(
                vis_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )

            # Display control values
            if action:
                cv2.putText(
                    vis_image,
                    f"x.vel: {action['x.vel']:.3f} m/s",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    vis_image,
                    f"y.vel: {action['y.vel']:.3f} m/s",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    vis_image,
                    f"theta.vel: {action['theta.vel']:.1f} deg/s",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Display additional info for PD Control state
            if state_info.get("type") == "pd_control":
                lateral_error = state_info["lateral_error"]
                cv2.putText(
                    vis_image,
                    f"Lateral: {lateral_error:.1f}px",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    vis_image,
                    f"Lookahead: {LOOKAHEAD_DISTANCE:.0f}px",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            # Display smoothing status
            smoothing_status = f"Smoothing: {aligner.line_detector.detection_count}/{aligner.line_detector.min_confidence}"
            smoothing_color = (
                (0, 255, 0)
                if aligner.line_detector.detection_count >= aligner.line_detector.min_confidence
                else (255, 165, 0)
            )
            cv2.putText(
                vis_image,
                smoothing_status,
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                smoothing_color,
                2,
            )

            # Display horizontal line count
            h_count_text = f"H-Lines: {len(horizontal_polylines)}"
            h_count_color = (255, 0, 0) if horizontal_polylines else (128, 128, 128)
            cv2.putText(
                vis_image,
                h_count_text,
                (10, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                h_count_color,
                2,
            )

            # Display diagonal line detection info
            if len(diagonal_polylines) > 0:
                # Count left and right diagonal lines
                image_center = width / 2
                left_diag = sum(1 for polyline in diagonal_polylines
                               if sum(pt[0] for pt in polyline) / len(polyline) < image_center)
                right_diag = len(diagonal_polylines) - left_diag

                # Check if pair detected and get direction
                pair_detected, direction = aligner.detect_diagonal_pair(diagonal_polylines)

                # Display with different colors
                if pair_detected:
                    if direction == "up":
                        diag_text = f"D-Lines: {len(diagonal_polylines)} (L:{left_diag} R:{right_diag}) PAIR /\\"
                        diag_color = (255, 0, 255)  # Magenta for up (arrival)
                    elif direction == "down":
                        diag_text = f"D-Lines: {len(diagonal_polylines)} (L:{left_diag} R:{right_diag}) PAIR \\/"
                        diag_color = (0, 255, 255)  # Cyan for down (initial)
                    else:
                        diag_text = f"D-Lines: {len(diagonal_polylines)} (L:{left_diag} R:{right_diag}) PAIR ?"
                        diag_color = (0, 255, 0)  # Green for unknown
                else:
                    diag_text = f"D-Lines: {len(diagonal_polylines)} (L:{left_diag} R:{right_diag})"
                    diag_color = (200, 200, 200)  # Light gray for detected but no pair
            else:
                diag_text = f"D-Lines: 0"
                diag_color = (128, 128, 128)  # Gray for no detection

            cv2.putText(
                vis_image,
                diag_text,
                (10, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                diag_color,
                2,
            )

            # Display test mode
            test_mode_text = f"Mode: {TEST_MODE}"
            test_mode_color = (0, 255, 255) if TEST_MODE != "all" else (200, 200, 200)  # Yellow for test modes
            cv2.putText(
                vis_image,
                test_mode_text,
                (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                test_mode_color,
                2,
            )

            fps_text = f"Control: {displayed_fps:.1f} Hz"
            fps_color = (0, 255, 0) if displayed_fps >= CONTROL_HZ * 0.9 else (0, 165, 255)  # Green if close to target, orange otherwise
            cv2.putText(
                vis_image,
                fps_text,
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                fps_color,
                2,
            )

            t_vis_end = time.perf_counter()

            # Calculate timing values after all measurements are complete
            elapsed = time.perf_counter() - start_time
            actual_fps = 1.0 / max(elapsed, 1e-6)

            t_obs_ms = (t_obs_end - t_obs_start) * 1000
            t_detect_ms = (t_detect_end - t_detect_start) * 1000
            t_control_ms = (t_control_end - t_control_start) * 1000
            t_action_ms = (t_action_end - t_action_start) * 1000
            t_vis_ms = (t_vis_end - t_vis_start) * 1000
            t_total_ms = elapsed * 1000

            # Collect samples
            fps_samples.append(actual_fps)
            timing_samples['obs'].append(t_obs_ms)
            timing_samples['detect'].append(t_detect_ms)
            timing_samples['control'].append(t_control_ms)
            timing_samples['action'].append(t_action_ms)
            timing_samples['visual'].append(t_vis_ms)
            timing_samples['total'].append(t_total_ms)

            # Update displayed values every 0.3 seconds by averaging samples
            current_time = time.perf_counter()
            if current_time - last_display_update_time >= display_update_interval:
                # Update FPS
                if fps_samples:
                    displayed_fps = sum(fps_samples) / len(fps_samples)
                    fps_samples = []

                # Update timing values
                for key in displayed_timing.keys():
                    if timing_samples[key]:
                        displayed_timing[key] = sum(timing_samples[key]) / len(timing_samples[key])
                        timing_samples[key] = []

                last_display_update_time = current_time

            # Display timing breakdown for performance analysis
            timing_y_start = 360
            timing_line_height = 25

            cv2.putText(vis_image, "=== Timing (ms) ===", (10, timing_y_start),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(vis_image, f"Get Obs: {displayed_timing['obs']:.1f}", (10, timing_y_start + timing_line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_image, f"Detect: {displayed_timing['detect']:.1f}", (10, timing_y_start + timing_line_height * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_image, f"Control: {displayed_timing['control']:.1f}", (10, timing_y_start + timing_line_height * 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_image, f"Action: {displayed_timing['action']:.1f}", (10, timing_y_start + timing_line_height * 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_image, f"Visual: {displayed_timing['visual']:.1f}", (10, timing_y_start + timing_line_height * 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis_image, f"Total: {displayed_timing['total']:.1f}", (10, timing_y_start + timing_line_height * 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            t_display_start = time.perf_counter()

            # Display result
            cv2.imshow("LeKiwi Line Following with State Machine", vis_image)

            t_display_end = time.perf_counter()

            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit signal received")
                break
            elif key == ord('n'):
                # 'n' 키: ARRIVED/INITIAL 상태에서 재개 시작
                if aligner.state == FollowState.ARRIVED:
                    # ARRIVED → 왼쪽으로 수평 이동 후 130° 회전 시작
                    aligner.action_phase = "lateral"
                    aligner.lateral_distance_traveled = 0.0
                    aligner.turn_direction = -1.0  # 회전 시 오른쪽
                    logger.info(f"'n' key pressed in ARRIVED state. Starting LEFT lateral ({LATERAL_DISTANCE_ON_RESUME:.2f}m) then RIGHT turn ({TURN_ANGLE_ON_RESUME:.1f}°).")
                elif aligner.state == FollowState.INITIAL:
                    # INITIAL → 오른쪽으로 130도 회전 시작
                    aligner.action_phase = "turn"
                    aligner.angle_turned = 0.0
                    aligner.turn_direction = -1.0  # 오른쪽 회전
                    logger.info(f"'n' key pressed in INITIAL state. Starting RIGHT turn ({TURN_ANGLE_ON_INITIAL:.1f}°) before resuming.")

            # Maintain control loop rate
            sleep_time = max(dt - elapsed, 0.0)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Log timing breakdown occasionally for performance analysis
            if int(start_time * 10) % 30 == 0:  # Log every ~3 seconds
                t_display_ms = (t_display_end - t_display_start) * 1000
                logger.info(
                    f"FPS: {displayed_fps:.1f} Hz | "
                    f"Timing (ms) - Obs: {displayed_timing['obs']:.1f}, Detect: {displayed_timing['detect']:.1f}, "
                    f"Control: {displayed_timing['control']:.1f}, Action: {displayed_timing['action']:.1f}, "
                    f"Visual: {displayed_timing['visual']:.1f}, Display: {t_display_ms:.1f}, "
                    f"Total: {displayed_timing['total']:.1f}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop robot
        aligner.stop()
        cv2.destroyAllWindows()
        robot.disconnect()
        logger.info("Disconnected from robot")


if __name__ == "__main__":
    main()
