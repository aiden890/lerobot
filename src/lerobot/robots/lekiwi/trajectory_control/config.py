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

"""Configuration loading module for trajectory control."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class DatasetConfig:
    """HuggingFace dataset configuration."""

    approach_pick: str
    approach_place: str


@dataclass
class RobotConfig:
    """Robot connection configuration."""

    remote_ip: str
    fps: float
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556


@dataclass
class CameraConfig:
    """Camera configuration."""

    name: str = "front"
    obs_key: str = "observation.images.front"
    type: str = "opencv"
    index_or_path: str = "/dev/video0"
    width: int = 640
    height: int = 480
    color_format: str = "rgb"  # "rgb" or "bgr"


@dataclass
class DirectionCalibrationPoint:
    """Direction calibration point: pixel_x -> shoulder_pan value."""

    pixel_x: int
    shoulder_pan: float


@dataclass
class CalibrationConfig:
    """Camera calibration configuration with robot origin offset."""

    # Robot origin is below camera view by this many pixels
    robot_origin_offset_px: int
    # Image center X coordinate
    image_center_x: int
    # Y coordinate of direction calibration points
    direction_reference_y: int
    # Direction calibration points (pixel_x -> shoulder_pan)
    direction_points: list[DirectionCalibrationPoint] = field(default_factory=list)


@dataclass
class DistancePoint:
    """Distance calibration point: timestep and pixel distance."""

    timestep: int
    pixel_distance: float


@dataclass
class ActionDistanceConfig:
    """Distance calibration for a single action (pick or place)."""

    near: DistancePoint
    far: DistancePoint


@dataclass
class DistanceConfig:
    """Distance calibration configuration for pick/place actions."""

    pick: ActionDistanceConfig
    place: ActionDistanceConfig


@dataclass
class MotionConfig:
    """Motion control configuration."""

    return_steps: int
    gripper_open: float
    gripper_close: float
    gripper_duration: float = 1.0
    shoulder_pan_speed: float = 30.0  # Shoulder pan movement speed in degrees per second


@dataclass
class ColorRange:
    """HSV color range for detection."""

    lower: list[int]
    upper: list[int]
    kernel_size: int = 5  # Morphological kernel size (smaller = detect smaller objects)


@dataclass
class ROIConfig:
    """Region of Interest configuration (normalized coordinates 0.0-1.0)."""

    pink: list[float] | None = None  # [x, y, width, height] for pink detection
    black: list[float] | None = None  # [x, y, width, height] for black detection


@dataclass
class ColorDetectionConfig:
    """Color detection configuration."""

    pink: ColorRange
    black: ColorRange
    min_area: int


@dataclass
class Config:
    """Complete configuration for trajectory control."""

    datasets: DatasetConfig
    robot: RobotConfig
    camera: CameraConfig
    calibration: CalibrationConfig
    distance: DistanceConfig
    motion: MotionConfig
    color_detection: ColorDetectionConfig
    roi: ROIConfig


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Uses default if None.

    Returns:
        Config object with all settings.
    """
    path = config_path or CONFIG_PATH

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse calibration config
    cal_data = data["calibration"]
    direction_points = [
        DirectionCalibrationPoint(pixel_x=p["pixel_x"], shoulder_pan=p["shoulder_pan"])
        for p in cal_data["direction"]["points"]
    ]
    calibration_config = CalibrationConfig(
        robot_origin_offset_px=cal_data["robot_origin_offset_px"],
        image_center_x=cal_data["image_center_x"],
        direction_reference_y=cal_data["direction"]["reference_y"],
        direction_points=direction_points,
    )

    # Parse camera config (with defaults if not present)
    camera_data = data.get("camera", {})
    camera_config = CameraConfig(
        name=camera_data.get("name", "front"),
        obs_key=camera_data.get("obs_key", "observation.images.front"),
        type=camera_data.get("type", "opencv"),
        index_or_path=camera_data.get("index_or_path", "/dev/video0"),
        width=camera_data.get("width", 640),
        height=camera_data.get("height", 480),
        color_format=camera_data.get("color_format", "rgb"),
    )

    # Parse distance calibration config
    dist_data = data.get("distance", {})
    pick_data = dist_data.get("pick", {})
    place_data = dist_data.get("place", {})

    def parse_distance_point(d: dict) -> DistancePoint:
        return DistancePoint(
            timestep=d.get("timestep", 0),
            pixel_distance=d.get("pixel_distance", 0.0),
        )

    distance_config = DistanceConfig(
        pick=ActionDistanceConfig(
            near=parse_distance_point(pick_data.get("near", {})),
            far=parse_distance_point(pick_data.get("far", {})),
        ),
        place=ActionDistanceConfig(
            near=parse_distance_point(place_data.get("near", {})),
            far=parse_distance_point(place_data.get("far", {})),
        ),
    )

    # Parse ROI config
    roi_data = data.get("roi", {})
    roi_config = ROIConfig(
        pink=roi_data.get("pink"),
        black=roi_data.get("black"),
    )

    return Config(
        datasets=DatasetConfig(**data["datasets"]),
        robot=RobotConfig(**data["robot"]),
        camera=camera_config,
        calibration=calibration_config,
        distance=distance_config,
        motion=MotionConfig(**data["motion"]),
        color_detection=ColorDetectionConfig(
            pink=ColorRange(**data["color_detection"]["pink"]),
            black=ColorRange(**data["color_detection"]["black"]),
            min_area=data["color_detection"]["min_area"],
        ),
        roi=roi_config,
    )


def get_config() -> Config:
    """Get the global configuration instance."""
    return load_config()
