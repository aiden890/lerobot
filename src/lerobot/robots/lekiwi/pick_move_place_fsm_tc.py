"""Trajectory Control based FSM controller for lekiwi pick-move-place routines.

This version uses pre-recorded trajectories and color detection instead of
policy inference for pick/place operations.
"""

import cv2
import logging
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

import draccus
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.utils.robot_utils import busy_wait

from .config_lekiwi import LeKiwiClientConfig
from .lekiwi_client import LeKiwiClient
from .follow_line import (
    VerticalLineAligner,
    FollowState,
    TURN_SPEED,
    TURN_SPEED_ON_INITIAL,
    TURN_SPEED_ON_RESUME,
    LATERAL_SPEED_ON_DIAGONAL,
    LATERAL_SPEED_ON_RESUME,
)
from .trajectory_control.action_generator import (
    TrajectoryActionGenerator,
    TrajectoryActionGeneratorConfig,
    TrajectoryPhase,
)
from .trajectory_control.camera_calibration import pixel_to_robot
from .trajectory_control.color_detection import (
    convert_to_bgr,
    detect_pink,
    detect_black,
)
from .trajectory_control.config import get_config as get_tc_config
from .trajectory_control.trajectory_utils import load_trajectory_from_dataset


class PickPlaceState(Enum):
    """Finite-state machine for pick → move → place."""

    IDLE = auto()
    PICK = auto()
    GOTO_PLACE = auto()
    PLACE = auto()
    GOTO_PICK = auto()


class PickPlaceCommand(Enum):
    START = auto()
    STOP = auto()


class PickPlaceController(Protocol):
    """Unified interface exposing both arm and base behaviours."""

    def start_pick(self) -> None:
        ...

    def pick_complete(self) -> bool:
        ...

    def start_place(self) -> None:
        ...

    def place_complete(self) -> bool:
        ...

    def follow_path(self, path_id: str) -> None:
        ...

    def motion_done(self) -> bool:
        ...

    def cancel(self) -> None:
        ...

    def build_action(self) -> dict[str, float] | None:
        ...

    def update_observation(self, observation: dict[str, Any]) -> None:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass
class PickPlaceConfig:
    pick_to_place_path: str = "pick_to_place"
    place_to_pick_path: str = "place_to_pick"


@dataclass
class LeKiwiPickPlaceFSM:
    """Minimal skeleton for the lekiwi pick-move-place controller."""

    controller: PickPlaceController
    config: PickPlaceConfig = field(default_factory=PickPlaceConfig)
    infinite_loop: bool = False
    require_manual_transition: bool = True  # Require 'n' key to transition between states

    state: PickPlaceState = PickPlaceState.IDLE
    _pending_command: PickPlaceCommand | None = None
    _force_next_state_event: threading.Event = field(default_factory=threading.Event)
    _ready_for_transition: bool = False  # True when current action is complete and waiting for 'n'

    def step(self, command: PickPlaceCommand | None = None) -> PickPlaceState:
        """Advance the FSM by one tick and return the current state."""

        if command:
            self._pending_command = command

        handler = getattr(self, f"_handle_{self.state.name.lower()}")
        handler()
        return self.state

    def force_next_state(self) -> None:
        """Force transition to next state, skipping completion checks."""
        self._force_next_state_event.set()

    def _check_force_next_state(self) -> bool:
        """Check and clear force next state flag (thread-safe)."""
        if self._force_next_state_event.is_set():
            self._force_next_state_event.clear()
            return True
        return False

    def is_waiting_for_input(self) -> bool:
        """Check if FSM is waiting for 'n' key to transition."""
        return self._ready_for_transition

    # State handlers -----------------------------------------------------
    def _handle_idle(self) -> None:
        if self._pending_command == PickPlaceCommand.START:
            self._pending_command = None
            self.controller.start_pick()
            self.state = PickPlaceState.PICK
            self._ready_for_transition = False
        elif self._pending_command == PickPlaceCommand.STOP:
            self._pending_command = None
        elif self._check_force_next_state():
            logging.info("Starting PICK from IDLE state")
            self.controller.start_pick()
            self.state = PickPlaceState.PICK
            self._ready_for_transition = False

    def _handle_pick(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return

        # Check if pick is complete
        if not self._ready_for_transition and self.controller.pick_complete():
            self._ready_for_transition = True
            logging.info("Pick complete - press 'n' to continue to GOTO_PLACE")

        # Handle transition
        if self._check_force_next_state():
            if self._ready_for_transition:
                logging.info("Transitioning to GOTO_PLACE")
            else:
                logging.info("Force skipping PICK state")
            self.controller.follow_path(self.config.pick_to_place_path)
            self.state = PickPlaceState.GOTO_PLACE
            self._ready_for_transition = False
            return

        # Auto-transition if manual transition not required
        if not self.require_manual_transition and self._ready_for_transition:
            logging.info("Pick complete, auto-transitioning to GOTO_PLACE")
            self.controller.follow_path(self.config.pick_to_place_path)
            self.state = PickPlaceState.GOTO_PLACE
            self._ready_for_transition = False

    def _handle_goto_place(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return

        # Check if motion is complete
        if not self._ready_for_transition and self.controller.motion_done():
            self._ready_for_transition = True
            logging.info("GOTO_PLACE complete - press 'n' to continue to PLACE")

        # Handle transition
        if self._check_force_next_state():
            if self._ready_for_transition:
                logging.info("Transitioning to PLACE")
            else:
                logging.info("Force skipping GOTO_PLACE state")
            self.controller.start_place()
            self.state = PickPlaceState.PLACE
            self._ready_for_transition = False
            return

        # Auto-transition if manual transition not required
        if not self.require_manual_transition and self._ready_for_transition:
            logging.info("GOTO_PLACE complete, auto-transitioning to PLACE")
            self.controller.start_place()
            self.state = PickPlaceState.PLACE
            self._ready_for_transition = False

    def _handle_place(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return

        # Check if place is complete
        if not self._ready_for_transition and self.controller.place_complete():
            self._ready_for_transition = True
            logging.info("Place complete - press 'n' to continue to GOTO_PICK")

        # Handle transition
        if self._check_force_next_state():
            if self._ready_for_transition:
                logging.info("Transitioning to GOTO_PICK")
            else:
                logging.info("Force skipping PLACE state")
            self.controller.follow_path(self.config.place_to_pick_path)
            self.state = PickPlaceState.GOTO_PICK
            self._ready_for_transition = False
            return

        # Auto-transition if manual transition not required
        if not self.require_manual_transition and self._ready_for_transition:
            logging.info("Place complete, auto-transitioning to GOTO_PICK")
            self.controller.follow_path(self.config.place_to_pick_path)
            self.state = PickPlaceState.GOTO_PICK
            self._ready_for_transition = False

    def _handle_goto_pick(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return

        # Check if motion is complete
        if not self._ready_for_transition and self.controller.motion_done():
            self._ready_for_transition = True
            logging.info("GOTO_PICK complete - press 'n' to continue")

        # Handle transition
        if self._check_force_next_state():
            if self._ready_for_transition:
                logging.info("Transitioning from GOTO_PICK")
            else:
                logging.info("Force skipping GOTO_PICK state")
            self._transition_to_idle_or_pick()
            self._ready_for_transition = False
            return

        # Auto-transition if manual transition not required
        if not self.require_manual_transition and self._ready_for_transition:
            logging.info("GOTO_PICK complete, auto-transitioning")
            self._transition_to_idle_or_pick()
            self._ready_for_transition = False

    def _transition_to_idle_or_pick(self) -> None:
        """Transition to IDLE or auto-start PICK based on infinite_loop setting."""
        if self.infinite_loop:
            logging.info("Infinite loop: starting PICK")
            self.controller.start_pick()
            self.state = PickPlaceState.PICK
        else:
            self.state = PickPlaceState.IDLE

    def _abort(self) -> None:
        self._pending_command = None
        self.controller.cancel()
        self.state = PickPlaceState.IDLE
        self._ready_for_transition = False


class TrajectoryControlPickPlaceController(PickPlaceController):
    """Controls both arm and base behaviour using trajectory control."""

    def __init__(self, robot: Robot, environment_dt: float = 1 / 30):
        self._robot = robot
        self._environment_dt = environment_dt
        self._logger = logging.getLogger("tc_controller")
        self._current_phase: str | None = None
        self._phase_lock = threading.Lock()
        self._path_done_event = threading.Event()
        self._path_done_event.set()
        self._action_keys = list(robot.action_features.keys())

        # Trajectory control config
        self._tc_config = get_tc_config()

        # Action generator
        generator_config = TrajectoryActionGeneratorConfig(
            fps=self._tc_config.robot.fps,
            return_steps=self._tc_config.motion.return_steps,
            gripper_open=self._tc_config.motion.gripper_open,
            gripper_close=self._tc_config.motion.gripper_close,
            gripper_duration=self._tc_config.motion.gripper_duration,
            shoulder_pan_speed=self._tc_config.motion.shoulder_pan_speed,
        )
        self._action_generator = TrajectoryActionGenerator(generator_config)

        # Load trajectories
        self._pick_trajectory: list[dict] = []
        self._place_trajectory: list[dict] = []
        self._initial_state: dict = {}

        # Line following
        self._line_aligner: VerticalLineAligner | None = None
        self._line_follow_target: FollowState | None = None
        self._is_line_following: bool = False
        self._line_follow_none_count: int = 0

        # Current observation
        self._current_observation: dict[str, Any] | None = None

    def _load_trajectories(self) -> None:
        """Load pick and place trajectories from HuggingFace datasets."""
        if self._pick_trajectory and self._place_trajectory:
            return  # Already loaded

        self._logger.info("Loading trajectories from HuggingFace...")

        # Load pick trajectory
        pick_dataset = self._tc_config.datasets.approach_pick
        self._logger.info(f"Loading pick trajectory from {pick_dataset}")
        self._pick_trajectory = load_trajectory_from_dataset(pick_dataset)
        self._logger.info(f"Pick trajectory loaded: {len(self._pick_trajectory)} frames")

        # Load place trajectory
        place_dataset = self._tc_config.datasets.approach_place
        self._logger.info(f"Loading place trajectory from {place_dataset}")
        self._place_trajectory = load_trajectory_from_dataset(place_dataset)
        self._logger.info(f"Place trajectory loaded: {len(self._place_trajectory)} frames")

    def _capture_initial_state(self) -> None:
        """Capture current robot state as initial state for returning."""
        obs = self._robot.get_observation()
        self._initial_state = {k: v for k, v in obs.items() if k.endswith(".pos")}

    def _pixel_distance_to_ratio(self, pixel_dist: float, action_type: str) -> float:
        """
        Convert pixel distance to trajectory ratio using distance calibration.

        Args:
            pixel_dist: Distance from robot origin in pixels
            action_type: "pick" or "place"

        Returns:
            Ratio of trajectory to play (0.0 to 1.0)
        """
        if action_type == "pick":
            cal = self._tc_config.distance.pick
            traj_len = len(self._pick_trajectory)
        else:
            cal = self._tc_config.distance.place
            traj_len = len(self._place_trajectory)

        near_dist = cal.near.pixel_distance
        far_dist = cal.far.pixel_distance
        near_ts = cal.near.timestep
        far_ts = cal.far.timestep

        # Linear interpolation of distance to timestep
        if far_dist == near_dist:
            target_ts = near_ts
        else:
            dist_ratio = (pixel_dist - near_dist) / (far_dist - near_dist)
            dist_ratio = max(0.0, min(1.0, dist_ratio))  # Clamp to [0, 1]
            target_ts = near_ts + dist_ratio * (far_ts - near_ts)

        # Convert timestep to trajectory ratio
        ratio = target_ts / (traj_len - 1) if traj_len > 1 else 0.0
        ratio = max(0.0, min(1.0, ratio))

        self._logger.debug(
            f"Distance calibration: pixel_dist={pixel_dist:.1f} -> target_ts={target_ts:.0f} -> ratio={ratio:.3f}"
        )
        return ratio

    def _check_distance_in_range(self, pixel_distance: float, action_type: str) -> bool:
        """Check if pixel distance is within calibration range.

        Args:
            pixel_distance: Detected pixel distance
            action_type: "pick" or "place"

        Returns:
            True if within range, False otherwise
        """
        if action_type == "pick":
            cal = self._tc_config.distance.pick
        else:
            cal = self._tc_config.distance.place

        near_dist = cal.near.pixel_distance
        far_dist = cal.far.pixel_distance
        min_dist = min(near_dist, far_dist)
        max_dist = max(near_dist, far_dist)

        if pixel_distance < min_dist or pixel_distance > max_dist:
            if pixel_distance < min_dist:
                self._logger.warning(
                    f"Object too CLOSE: {pixel_distance:.1f}px < {min_dist:.1f}px (min)"
                )
            else:
                self._logger.warning(
                    f"Object too FAR: {pixel_distance:.1f}px > {max_dist:.1f}px (max)"
                )
            return False

        return True

    def _detect_and_start_pick(self) -> bool:
        """Detect pink object and start pick trajectory."""
        if self._current_observation is None:
            self._logger.warning("No observation available for detection")
            return False

        # Get camera image
        image = self._current_observation.get(self._tc_config.camera.obs_key)
        if image is None:
            self._logger.warning(f"No camera image (key: {self._tc_config.camera.obs_key})")
            return False

        # Convert to BGR for detection
        image_bgr = convert_to_bgr(image, self._tc_config.camera.color_format)
        h, w = image_bgr.shape[:2]

        # Detect pink object
        result = detect_pink(image_bgr)
        if result is None:
            self._logger.warning("No pink object detected")
            return False

        cx, cy, _, _ = result
        pixel_distance, shoulder_pan = pixel_to_robot(cx, cy, h)

        # Check if within calibration range
        if not self._check_distance_in_range(pixel_distance, "pick"):
            return False

        target_ratio = self._pixel_distance_to_ratio(pixel_distance, "pick")

        self._logger.info(
            f"Pick target: ({cx}, {cy}) -> pixel_dist={pixel_distance:.1f}, "
            f"shoulder_pan={shoulder_pan:.1f}deg, ratio={target_ratio:.3f}"
        )

        # Start pick operation
        # After pick, return to first frame of place trajectory (not initial state)
        return_target = self._place_trajectory[0] if self._place_trajectory else None
        self._action_generator.start_pick(
            approach_trajectory=self._pick_trajectory,
            target_ratio=target_ratio,
            initial_state=self._initial_state,
            shoulder_pan_angle=shoulder_pan,
            return_target=return_target,
        )
        return True

    def _detect_and_start_place(self) -> bool:
        """Detect black region and start place trajectory."""
        if self._current_observation is None:
            self._logger.warning("No observation available for detection")
            return False

        # Get camera image
        image = self._current_observation.get(self._tc_config.camera.obs_key)
        if image is None:
            self._logger.warning(f"No camera image (key: {self._tc_config.camera.obs_key})")
            return False

        # Convert to BGR for detection
        image_bgr = convert_to_bgr(image, self._tc_config.camera.color_format)
        h, w = image_bgr.shape[:2]

        # Detect black region
        result = detect_black(image_bgr)
        if result is None:
            self._logger.warning("No black region detected")
            return False

        cx, cy, _, _ = result
        pixel_distance, shoulder_pan = pixel_to_robot(cx, cy, h)

        # Check if within calibration range
        if not self._check_distance_in_range(pixel_distance, "place"):
            return False

        target_ratio = self._pixel_distance_to_ratio(pixel_distance, "place")

        self._logger.info(
            f"Place target: ({cx}, {cy}) -> pixel_dist={pixel_distance:.1f}, "
            f"shoulder_pan={shoulder_pan:.1f}deg, ratio={target_ratio:.3f}"
        )

        # Start place operation
        # After place, return to first frame of pick trajectory (not initial state)
        return_target = self._pick_trajectory[0] if self._pick_trajectory else None
        self._action_generator.start_place(
            approach_trajectory=self._place_trajectory,
            target_ratio=target_ratio,
            initial_state=self._initial_state,
            shoulder_pan_angle=shoulder_pan,
            return_target=return_target,
        )
        return True

    def start_pick(self) -> None:
        """Start pick phase with trajectory control.

        Note: Detection is NOT performed here. The build_action() method will
        continuously attempt detection until a pink object is found.
        """
        self._load_trajectories()
        self._capture_initial_state()
        self._path_done_event.set()
        self._set_phase("pick")
        self._action_generator.reset()
        self._logger.info("Pick phase started - waiting for pink object detection")

    def pick_complete(self) -> bool:
        """Check if pick operation is complete."""
        return self._action_generator.is_done()

    def start_place(self) -> None:
        """Start place phase with trajectory control.

        Note: Detection is NOT performed here. The build_action() method will
        continuously attempt detection until a black region is found.
        """
        self._load_trajectories()
        self._capture_initial_state()
        self._path_done_event.set()
        self._set_phase("place")
        self._action_generator.reset()
        self._logger.info("Place phase started - waiting for black region detection")

    def place_complete(self) -> bool:
        """Check if place operation is complete."""
        return self._action_generator.is_done()

    def follow_path(self, path_id: str) -> None:
        """Start executing a named base path using line following."""
        if path_id == "pick_to_place":
            self._start_line_following(target_state=FollowState.ARRIVED, needs_initial_turn=True)
            self._logger.info("Starting line following to ARRIVED (pick_to_place) with initial 130° turn")
            return
        elif path_id == "place_to_pick":
            self._start_line_following(target_state=FollowState.INITIAL, needs_lateral_turn=True)
            self._logger.info("Starting line following to INITIAL (place_to_pick) with lateral left + 130° turn")
            return

        self._logger.warning("Unknown path '%s'", path_id)
        self._path_done_event.set()

    def motion_done(self) -> bool:
        if self._is_line_following and self._line_aligner:
            if self._line_aligner.state == self._line_follow_target:
                self._is_line_following = False
                self._path_done_event.set()
                self._logger.info("Line following complete: reached %s", self._line_follow_target.name)
                return True
            return False
        return self._path_done_event.is_set()

    def cancel(self) -> None:
        """Abort current activity."""
        self._logger.info("Cancelling current phase")
        self._set_phase(None)
        self._path_done_event.set()
        self._action_generator.reset()

    def build_action(self) -> dict[str, float] | None:
        """Build action based on current phase."""
        phase = self._current_phase
        if not phase:
            return None

        if phase == "pick" or phase == "place":
            if self._current_observation is None:
                return None

            # If generator is IDLE, try to detect and start trajectory
            if self._action_generator.is_idle():
                if phase == "pick":
                    detected = self._detect_and_start_pick()
                else:
                    detected = self._detect_and_start_place()

                if not detected:
                    # Object not detected - hold current position
                    return self._build_hold_position_action()

            # Get action from trajectory generator
            action = self._action_generator.get_action(self._current_observation)
            if action is not None:
                return self._normalize_action_dict(action)
            return None

        # Line following phases
        if self._is_line_following and self._line_aligner:
            action = self._build_line_follow_action()
            if action is None:
                self._line_follow_none_count += 1
                if self._line_follow_none_count % 30 == 1:
                    self._logger.warning("Line follow action is None! (count: %d)", self._line_follow_none_count)
            else:
                self._line_follow_none_count = 0
            return action

        return None

    def update_observation(self, observation: dict[str, Any]) -> None:
        """Update current observation."""
        self._current_observation = observation

    def start(self) -> None:
        """Start controller."""
        self._load_trajectories()

    def stop(self) -> None:
        """Stop controller."""
        self._action_generator.reset()
        self._current_observation = None

    def _set_phase(self, phase: str | None) -> None:
        with self._phase_lock:
            self._current_phase = phase

    def _normalize_action_dict(self, action: dict[str, float]) -> dict[str, float]:
        return {key: float(action.get(key, 0.0)) for key in self._action_keys}

    def _build_hold_position_action(self) -> dict[str, float] | None:
        """Build an action that holds the robot in its current position."""
        if self._current_observation is None:
            return None

        action = {}
        for key in self._action_keys:
            if key.endswith(".pos"):
                # Keep current position
                action[key] = self._current_observation.get(key, 0.0)
            elif key.endswith(".vel"):
                # Zero velocity
                action[key] = 0.0
            else:
                action[key] = 0.0

        return action

    # =========================================================================
    # Line Following Methods (from LocalInferencePickPlaceController)
    # =========================================================================

    def _start_line_following(
        self, target_state: FollowState, needs_initial_turn: bool = False, needs_lateral_turn: bool = False
    ) -> None:
        """Initialize line following towards the target state."""
        self._logger.info(
            "_start_line_following called with target=%s, needs_initial_turn=%s, needs_lateral_turn=%s",
            target_state.name,
            needs_initial_turn,
            needs_lateral_turn,
        )

        if self._line_aligner is None:
            self._line_aligner = VerticalLineAligner(self._robot)
            self._logger.info("Created new VerticalLineAligner")

        if needs_initial_turn:
            self._line_aligner.state = FollowState.INITIAL_TURN
            self._line_aligner.angle_turned = 0.0
            self._line_aligner.action_phase = "forward"
            self._logger.info("Starting with INITIAL_TURN state (130° turn before line following)")
        elif needs_lateral_turn:
            self._line_aligner.state = FollowState.ARRIVED
            self._line_aligner.action_phase = "lateral"
            self._line_aligner.lateral_distance_traveled = 0.0
            self._line_aligner.angle_turned = 0.0
            self._line_aligner.turn_direction = -1.0
            self._logger.info("Starting with ARRIVED state (lateral left → 130° turn before line following)")
        else:
            self._line_aligner.state = FollowState.FOLLOW_VERTICAL
            self._line_aligner.action_phase = "forward"

        self._line_aligner.horizontal_detected = False
        self._line_aligner.diagonal_detected = False
        self._line_aligner.distance_traveled = 0.0
        if not needs_lateral_turn:
            self._line_aligner.lateral_distance_traveled = 0.0
            self._line_aligner.turn_direction = -1.0

        self._line_follow_target = target_state
        self._is_line_following = True
        self._path_done_event.clear()
        self._set_phase(f"line_follow_{target_state.name.lower()}")
        self._logger.info(
            "Line following initialized: _is_line_following=%s, aligner_state=%s, phase=%s",
            self._is_line_following,
            self._line_aligner.state.name,
            self._current_phase,
        )

    def _build_line_follow_action(self) -> dict[str, float] | None:
        """Build action for line following based on camera input."""
        if not self._line_aligner:
            return None

        dt = self._environment_dt

        # Get camera image from observation
        observation = self._robot.get_observation()
        camera_name = list(self._robot.config.cameras.keys())[0]
        image = observation[camera_name]
        height, width = image.shape[:2]

        # Detect lines
        vertical, horizontal, diagonal = self._line_aligner.line_detector.detect_and_smooth(image)

        # Update state machine
        self._line_aligner.update_state(horizontal, diagonal, width, height, dt)

        # Check if target reached
        if self._line_aligner.state == self._line_follow_target:
            self._is_line_following = False
            self._path_done_event.set()
            self._logger.info("Line following complete: reached %s", self._line_follow_target.name)
            return None

        return self._compute_line_action(vertical, width, height)

    def _compute_line_action(
        self, vertical_polylines: list, image_width: int, image_height: int
    ) -> dict[str, float] | None:
        """Compute action based on the current line aligner state."""
        aligner = self._line_aligner
        if not aligner:
            return None

        action = None

        if aligner.state == FollowState.INITIAL_TURN:
            action = aligner.compute_turn_control(TURN_SPEED_ON_RESUME)

        elif aligner.state == FollowState.FOLLOW_VERTICAL:
            if vertical_polylines:
                result = aligner.compute_pd_control(vertical_polylines, image_width, image_height)
                if result is not None:
                    action, _, _, _, _ = result

        elif aligner.state == FollowState.HORIZONTAL_ACTION:
            if aligner.action_phase == "forward":
                action = aligner.compute_forward_control()
            elif aligner.action_phase == "turn":
                action = aligner.compute_turn_control(TURN_SPEED)

        elif aligner.state == FollowState.DIAGONAL_ACTION:
            if aligner.action_phase == "forward":
                action = aligner.compute_forward_control()
            elif aligner.action_phase == "lateral":
                action = aligner.compute_lateral_control(LATERAL_SPEED_ON_DIAGONAL, direction=-1.0)

        elif aligner.state == FollowState.INITIAL_ACTION:
            action = aligner.compute_forward_control()

        elif aligner.state == FollowState.ARRIVED:
            if aligner.action_phase == "lateral":
                action = aligner.compute_lateral_control(LATERAL_SPEED_ON_RESUME, direction=1.0)
            elif aligner.action_phase == "turn":
                action = aligner.compute_turn_control(TURN_SPEED_ON_RESUME)
            else:
                action = None

        elif aligner.state == FollowState.INITIAL:
            if aligner.action_phase == "turn":
                action = aligner.compute_turn_control(TURN_SPEED_ON_INITIAL)
            else:
                action = None

        return action


@dataclass
class TCPickPlaceLoopConfig:
    robot: LeKiwiClientConfig = field(default_factory=LeKiwiClientConfig)
    fsm: PickPlaceConfig = field(default_factory=PickPlaceConfig)
    target_hz: float = 30.0
    auto_start: bool = True
    infinite_loop: bool = True
    enable_keyboard_control: bool = True
    require_manual_transition: bool = True  # Require 'n' key to transition between states


def _keyboard_listener(fsm: LeKiwiPickPlaceFSM, shutdown_event: threading.Event) -> None:
    """Listen for keyboard input to force state transitions."""
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        logging.info("Keyboard control enabled. Press 'n' to skip to next state, 'q' to quit.")

        while not shutdown_event.is_set():
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == "n":
                    logging.info("User requested force next state (current: %s)", fsm.state.name)
                    fsm.force_next_state()
                elif key == "q":
                    logging.info("User requested quit")
                    fsm.step(PickPlaceCommand.STOP)
                    shutdown_event.set()
                    break
    except Exception as err:
        logging.error("Keyboard listener error: %s", err)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def _run_pick_place_control_loop(
    robot: Robot,
    controller: TrajectoryControlPickPlaceController,
    fsm: LeKiwiPickPlaceFSM,
    target_hz: float,
    shutdown_event: threading.Event,
) -> None:
    timestep = 0
    period = 1.0 / max(target_hz, 1e-3)

    # FPS tracking
    fps_window_size = 30
    loop_times = []
    last_print_time = time.perf_counter()
    print_interval = 0.5

    while not shutdown_event.is_set():
        loop_start = time.perf_counter()

        # FSM step
        fsm.step()

        # Get observation
        observation = robot.get_observation()

        # Update controller with observation
        controller.update_observation(observation)

        # Build action
        action = controller.build_action()

        # Send action
        if action:
            robot.send_action(action)

        timestep += 1

        elapsed = time.perf_counter() - loop_start
        busy_wait(max(0.0, period - elapsed))

        # Track FPS
        total_time = time.perf_counter() - loop_start
        loop_times.append(total_time)
        if len(loop_times) > fps_window_size:
            loop_times.pop(0)

        # Print FPS info
        current_time = time.perf_counter()
        if current_time - last_print_time >= print_interval:
            avg_loop_time = sum(loop_times) / len(loop_times)
            current_fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

            # Get action generator progress
            progress = controller._action_generator.get_progress()
            phase_info = f"{progress['phase']}({progress['progress']:.0%})" if progress["phase"] != "IDLE" else "IDLE"

            # Show waiting status
            wait_status = " [WAIT 'n']" if fsm.is_waiting_for_input() else ""

            print(
                f"\rFPS: {current_fps:.1f}/{target_hz:.0f} | "
                f"FSM: {fsm.state.name}{wait_status} | TC: {phase_info} | Step: {timestep}    ",
                end="",
                flush=True,
            )
            last_print_time = current_time


@draccus.wrap()
def pick_place_tc_control_loop(cfg: TCPickPlaceLoopConfig) -> None:
    logging.info("Starting pick-place control loop (TRAJECTORY CONTROL)")

    robot = LeKiwiClient(cfg.robot)
    robot.connect()
    controller = TrajectoryControlPickPlaceController(robot, environment_dt=1.0 / cfg.target_hz)
    fsm = LeKiwiPickPlaceFSM(
        controller=controller,
        config=cfg.fsm,
        infinite_loop=cfg.infinite_loop,
        require_manual_transition=cfg.require_manual_transition,
    )

    if cfg.auto_start:
        fsm.step(PickPlaceCommand.START)

    shutdown_event = threading.Event()
    keyboard_thread = None

    try:
        controller.start()

        if cfg.enable_keyboard_control:
            keyboard_thread = threading.Thread(
                target=_keyboard_listener,
                args=(fsm, shutdown_event),
                name="keyboard_listener",
                daemon=True,
            )
            keyboard_thread.start()

        _run_pick_place_control_loop(robot, controller, fsm, target_hz=cfg.target_hz, shutdown_event=shutdown_event)
    except KeyboardInterrupt:
        logging.info("Stopping pick-place loop (KeyboardInterrupt)")
        shutdown_event.set()
    finally:
        shutdown_event.set()
        if keyboard_thread and keyboard_thread.is_alive():
            keyboard_thread.join(timeout=1.0)
        controller.stop()
        robot.disconnect()


def main() -> None:
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fsm_tc_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logging.info(f"Log file: {log_file}")
    logging.info("Running TRAJECTORY CONTROL mode")
    pick_place_tc_control_loop()


if __name__ == "__main__":
    main()
