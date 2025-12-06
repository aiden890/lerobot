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
Trajectory-based Pick/Place Control Module

This module provides vision-based pick and place functionality using
pre-recorded trajectories with distance-proportional playback.
"""

from .camera_calibration import (
    draw_calibration_overlay,
    draw_calibration_points,
    get_detection_info,
    pixel_to_real,
    pixel_to_robot,
    real_to_normalized,
    reload_calibration,
    visualize_detection,
)
from .color_detection import (
    COLOR_RANGES,
    TargetColor,
    detect_black,
    detect_color_region,
    detect_pink,
)
from .config import Config, load_config
from .trajectory_calibration import (
    DEFAULT_JOINT_LIMITS,
    check_trajectory_range,
    print_trajectory_info,
    validate_trajectory,
)
from .trajectory_utils import (
    load_trajectory_from_dataset,
    move_to_initial,
    play_full_trajectory,
    play_trajectory_until,
)
from .action_generator import (
    TrajectoryActionGenerator,
    TrajectoryActionGeneratorConfig,
    TrajectoryPhase,
)

__all__ = [
    # Camera calibration
    "draw_calibration_overlay",
    "draw_calibration_points",
    "get_detection_info",
    "pixel_to_real",
    "pixel_to_robot",
    "real_to_normalized",
    "reload_calibration",
    "visualize_detection",
    # Color detection
    "COLOR_RANGES",
    "TargetColor",
    "detect_black",
    "detect_color_region",
    "detect_pink",
    # Config
    "Config",
    "load_config",
    # Trajectory calibration
    "DEFAULT_JOINT_LIMITS",
    "check_trajectory_range",
    "print_trajectory_info",
    "validate_trajectory",
    # Trajectory utils
    "load_trajectory_from_dataset",
    "move_to_initial",
    "play_full_trajectory",
    "play_trajectory_until",
    # Action generator
    "TrajectoryActionGenerator",
    "TrajectoryActionGeneratorConfig",
    "TrajectoryPhase",
]
