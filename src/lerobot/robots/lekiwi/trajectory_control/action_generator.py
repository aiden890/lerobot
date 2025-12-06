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

"""Action generator for FSM integration with trajectory-based pick/place."""

import time

from dataclasses import dataclass
from enum import Enum, auto


class TrajectoryPhase(Enum):
    """Phases of trajectory-based action execution."""

    IDLE = auto()  # Waiting for action start
    SHOULDER_PAN = auto()  # Moving shoulder pan to target angle first
    APPROACHING = auto()  # Playing approach trajectory
    GRIPPER_ACTION = auto()  # Opening or closing gripper
    RETURNING = auto()  # Returning to initial state
    DONE = auto()  # Action completed


@dataclass
class TrajectoryActionGeneratorConfig:
    """Configuration for TrajectoryActionGenerator."""

    fps: float = 30.0
    return_steps: int = 50  # Steps for returning to initial
    gripper_open: float = 100.0  # Gripper open position
    gripper_close: float = 0.0  # Gripper close position
    gripper_duration: float = 1.0  # Gripper action duration in seconds
    shoulder_pan_speed: float = 30.0  # Shoulder pan movement speed in degrees per second
    wrist_flex_lift_duration: float = 1.0  # Duration to lift wrist flex before returning (after place)
    wrist_flex_lift_angle: float = -40.0  # Angle to lift wrist flex (negative = up, after place)
    wrist_flex_place_offset: float = 10.0  # Offset to apply to wrist flex during place approach (positive = down)


class TrajectoryActionGenerator:
    """
    FSM-compatible action generator for trajectory-based pick/place operations.

    This class manages the state machine for executing pick or place operations
    using pre-recorded trajectories. It can be integrated into existing FSM systems.

    Usage:
        generator = TrajectoryActionGenerator()
        generator.start_pick(approach_traj, target_ratio, initial_state)

        while not generator.is_done():
            obs = robot.get_observation()
            action = generator.get_action(obs)
            if action:
                robot.send_action(action)
    """

    def __init__(self, config: TrajectoryActionGeneratorConfig | None = None):
        """
        Initialize the action generator.

        Args:
            config: Configuration settings. Uses defaults if None.
        """
        self.config = config or TrajectoryActionGeneratorConfig()
        self.phase = TrajectoryPhase.IDLE

        # Trajectory data
        self._approach_trajectory: list[dict] = []
        self._target_idx: int = 0
        self._current_idx: int = 0
        self._initial_state: dict = {}
        self._return_target: dict = {}  # Target state for returning phase
        self._is_pick: bool = True  # True = pick (close gripper), False = place (open gripper)

        # Shoulder pan angle for directional control during approach
        self._shoulder_pan_angle: float | None = None
        self._shoulder_pan_start_angle: float = 0.0
        self._shoulder_pan_steps: int = 0
        self._shoulder_pan_current_step: int = 0
        self._shoulder_pan_obs_at_start: dict = {}

        # Gripper action state
        self._gripper_start_time: float | None = None
        self._gripper_start_pos: float = 0.0
        self._gripper_obs_at_start: dict = {}

        # Return state
        self._return_step: int = 0
        self._return_start_obs: dict = {}

    def start_pick(
        self,
        approach_trajectory: list[dict],
        target_ratio: float,
        initial_state: dict,
        shoulder_pan_angle: float | None = None,
        return_target: dict | None = None,
    ) -> None:
        """
        Start a pick operation.

        Args:
            approach_trajectory: Trajectory for approaching the object (distance proportional)
            target_ratio: Ratio of approach trajectory to play (0.0 to 1.0)
            initial_state: State to return to after operation (used if return_target is None)
            shoulder_pan_angle: Angle in degrees for shoulder pan during approach (None to use trajectory values)
            return_target: Target state for returning phase (e.g., first frame of place trajectory).
                          If None, uses initial_state.
        """
        self._approach_trajectory = approach_trajectory
        self._target_idx = int(target_ratio * (len(approach_trajectory) - 1))
        self._current_idx = 0
        self._initial_state = initial_state.copy()
        self._return_target = return_target.copy() if return_target else initial_state.copy()
        self._is_pick = True
        self._shoulder_pan_angle = shoulder_pan_angle
        self._gripper_start_time = None

        # Start with shoulder pan phase if angle is specified, otherwise go directly to approaching
        if shoulder_pan_angle is not None:
            self.phase = TrajectoryPhase.SHOULDER_PAN
            self._shoulder_pan_current_step = 0
            self._shoulder_pan_steps = 0  # Will be calculated on first get_action call
        else:
            self.phase = TrajectoryPhase.APPROACHING

    def start_place(
        self,
        approach_trajectory: list[dict],
        target_ratio: float,
        initial_state: dict,
        shoulder_pan_angle: float | None = None,
        return_target: dict | None = None,
    ) -> None:
        """
        Start a place operation.

        Args:
            approach_trajectory: Trajectory for approaching the place position (distance proportional)
            target_ratio: Ratio of approach trajectory to play (0.0 to 1.0)
            initial_state: State to return to after operation (used if return_target is None)
            shoulder_pan_angle: Angle in degrees for shoulder pan during approach (None to use trajectory values)
            return_target: Target state for returning phase (e.g., first frame of pick trajectory).
                          If None, uses initial_state.
        """
        self._approach_trajectory = approach_trajectory
        self._target_idx = int(target_ratio * (len(approach_trajectory) - 1))
        self._current_idx = 0
        self._initial_state = initial_state.copy()
        self._return_target = return_target.copy() if return_target else initial_state.copy()
        self._is_pick = False
        self._shoulder_pan_angle = shoulder_pan_angle
        self._gripper_start_time = None

        # Start with shoulder pan phase if angle is specified, otherwise go directly to approaching
        if shoulder_pan_angle is not None:
            self.phase = TrajectoryPhase.SHOULDER_PAN
            self._shoulder_pan_current_step = 0
            self._shoulder_pan_steps = 0  # Will be calculated on first get_action call
        else:
            self.phase = TrajectoryPhase.APPROACHING

    def get_action(self, current_observation: dict) -> dict | None:
        """
        Get the next action based on current phase.

        Call this in your control loop to get the action to send to the robot.

        Args:
            current_observation: Current robot observation (from get_observation)

        Returns:
            Action dictionary to send to robot, or None if done
        """
        if self.phase == TrajectoryPhase.IDLE:
            return None

        if self.phase == TrajectoryPhase.SHOULDER_PAN:
            return self._handle_shoulder_pan(current_observation)

        if self.phase == TrajectoryPhase.APPROACHING:
            return self._handle_approaching(current_observation)

        if self.phase == TrajectoryPhase.GRIPPER_ACTION:
            return self._handle_gripper_action(current_observation)

        if self.phase == TrajectoryPhase.RETURNING:
            return self._handle_returning(current_observation)

        return None

    def _handle_shoulder_pan(self, current_observation: dict) -> dict | None:
        """Handle the shoulder pan movement phase (move shoulder pan first, then approach)."""
        # Initialize on first call
        if self._shoulder_pan_steps == 0:
            self._shoulder_pan_obs_at_start = current_observation.copy()
            self._shoulder_pan_start_angle = current_observation.get("arm_shoulder_pan.pos", 0.0)
            angle_diff = abs(self._shoulder_pan_angle - self._shoulder_pan_start_angle)

            if angle_diff < 0.5:  # Already at target
                self.phase = TrajectoryPhase.APPROACHING
                return self._handle_approaching(current_observation)

            # Calculate steps based on constant speed
            duration = angle_diff / self.config.shoulder_pan_speed
            self._shoulder_pan_steps = max(1, int(duration * self.config.fps))

        if self._shoulder_pan_current_step < self._shoulder_pan_steps:
            alpha = (self._shoulder_pan_current_step + 1) / self._shoulder_pan_steps

            # Build action from observation at start, only changing shoulder pan
            action = {k: v for k, v in self._shoulder_pan_obs_at_start.items() if k.endswith(".pos")}
            action["arm_shoulder_pan.pos"] = (
                self._shoulder_pan_start_angle * (1 - alpha) + self._shoulder_pan_angle * alpha
            )

            # Keep base velocity at zero
            for k in ["x.vel", "y.vel", "theta.vel"]:
                action[k] = 0.0

            self._shoulder_pan_current_step += 1
            return action
        else:
            # Shoulder pan complete -> start approaching
            self.phase = TrajectoryPhase.APPROACHING
            return self._handle_approaching(current_observation)

    def _handle_approaching(self, current_observation: dict) -> dict | None:
        """Handle the approaching phase."""
        if self._current_idx <= self._target_idx:
            action = self._approach_trajectory[self._current_idx].copy()

            # Keep shoulder pan fixed at target angle during trajectory playback
            # (shoulder pan was already moved in SHOULDER_PAN phase)
            if self._shoulder_pan_angle is not None and "arm_shoulder_pan.pos" in action:
                action["arm_shoulder_pan.pos"] = self._shoulder_pan_angle

            # Apply wrist flex offset during place approach (positive = down/lower)
            if not self._is_pick and "arm_wrist_flex.pos" in action:
                action["arm_wrist_flex.pos"] += self.config.wrist_flex_place_offset

            self._current_idx += 1
            return action
        else:
            # Approach complete -> start gripper action
            self.phase = TrajectoryPhase.GRIPPER_ACTION
            self._gripper_start_time = time.time()
            self._gripper_obs_at_start = current_observation.copy()
            self._gripper_start_pos = current_observation.get("arm_gripper.pos", 50.0)
            return self._handle_gripper_action(current_observation)

    def _handle_gripper_action(self, current_observation: dict) -> dict | None:
        """Handle the gripper open/close phase."""
        elapsed = time.time() - self._gripper_start_time
        alpha = min(elapsed / self.config.gripper_duration, 1.0)

        # Determine target gripper position
        target_gripper = self.config.gripper_close if self._is_pick else self.config.gripper_open

        # Build action from observation at gripper start, only changing gripper
        action = {k: v for k, v in self._gripper_obs_at_start.items() if k.endswith(".pos")}
        action["arm_gripper.pos"] = self._gripper_start_pos * (1 - alpha) + target_gripper * alpha

        # Keep base velocity at zero
        for k in ["x.vel", "y.vel", "theta.vel"]:
            action[k] = 0.0

        if alpha >= 1.0:
            # Gripper action complete -> start returning
            self.phase = TrajectoryPhase.RETURNING
            self._return_step = 0

        return action

    def _handle_returning(self, current_observation: dict) -> dict | None:
        """Handle the returning phase.

        Returns to _return_target (e.g., first frame of the other trajectory).
        - After pick: gripper stays closed (don't interpolate)
        - After place: first lift wrist flex only (1s), then move all to return_target
        """
        if self._return_step == 0:
            self._return_start_obs = current_observation.copy()

        lift_frames = int(self.config.fps * self.config.wrist_flex_lift_duration)

        if self._is_pick:
            # After pick: normal interpolation with gripper closed
            if self._return_step < self.config.return_steps:
                alpha = (self._return_step + 1) / self.config.return_steps
                action = self._interpolate(self._return_start_obs, self._return_target, alpha)
                action["arm_gripper.pos"] = self.config.gripper_close
                self._return_step += 1
                return action
            else:
                self.phase = TrajectoryPhase.DONE
                return None
        else:
            # After place: only lift wrist flex, then done
            if self._return_step < lift_frames:
                # Phase 1: Only lift wrist flex by specific angle, keep others at start
                lift_alpha = (self._return_step + 1) / lift_frames
                action = {}
                for key in self._return_start_obs:
                    if key == "arm_wrist_flex.pos":
                        start_val = self._return_start_obs.get(key, 0.0)
                        # Lift by specific angle from current position (negative = up)
                        end_val = start_val + self.config.wrist_flex_lift_angle
                        action[key] = start_val * (1 - lift_alpha) + end_val * lift_alpha
                    elif key.endswith(".pos"):
                        action[key] = self._return_start_obs.get(key, 0.0)
                    elif key.endswith(".vel"):
                        action[key] = 0.0

                self._return_step += 1
                return action
            else:
                # Phase 2 commented out - go directly to DONE after wrist flex lift
                # phase2_step = self._return_step - lift_frames
                # alpha = (phase2_step + 1) / self.config.return_steps
                # phase1_end = self._return_start_obs.copy()
                # phase1_end["arm_wrist_flex.pos"] = (
                #     self._return_start_obs.get("arm_wrist_flex.pos", 0.0) + self.config.wrist_flex_lift_angle
                # )
                # action = self._interpolate(phase1_end, self._return_target, alpha)
                self.phase = TrajectoryPhase.DONE
                return None

    def _interpolate(self, start: dict, end: dict, alpha: float) -> dict:
        """Linear interpolation between two states."""
        action = {}
        for key in end:
            if key.endswith(".pos"):
                start_val = start.get(key, end[key])
                action[key] = start_val * (1 - alpha) + end[key] * alpha
            elif key.endswith(".vel"):
                action[key] = 0.0
        return action

    def is_done(self) -> bool:
        """Check if the current operation is complete."""
        return self.phase == TrajectoryPhase.DONE

    def is_idle(self) -> bool:
        """Check if ready for a new operation."""
        return self.phase == TrajectoryPhase.IDLE

    def reset(self) -> None:
        """Reset the generator to idle state."""
        self.phase = TrajectoryPhase.IDLE
        self._approach_trajectory = []
        self._target_idx = 0
        self._current_idx = 0
        self._initial_state = {}
        self._return_target = {}
        self._shoulder_pan_angle = None
        self._shoulder_pan_start_angle = 0.0
        self._shoulder_pan_steps = 0
        self._shoulder_pan_current_step = 0
        self._shoulder_pan_obs_at_start = {}
        self._gripper_start_time = None
        self._gripper_start_pos = 0.0
        self._gripper_obs_at_start = {}
        self._return_step = 0
        self._return_start_obs = {}

    def get_progress(self) -> dict:
        """
        Get current progress information.

        Returns:
            Dictionary with phase, progress percentage, etc.
        """
        total_steps = 0
        current_step = 0

        if self.phase == TrajectoryPhase.SHOULDER_PAN:
            total_steps = self._shoulder_pan_steps if self._shoulder_pan_steps > 0 else 1
            current_step = self._shoulder_pan_current_step
        elif self.phase == TrajectoryPhase.APPROACHING:
            total_steps = self._target_idx + 1
            current_step = self._current_idx
        elif self.phase == TrajectoryPhase.GRIPPER_ACTION:
            # Gripper action progress based on time
            if self._gripper_start_time is not None:
                elapsed = time.time() - self._gripper_start_time
                total_steps = int(self.config.gripper_duration * self.config.fps)
                current_step = int(elapsed * self.config.fps)
        elif self.phase == TrajectoryPhase.RETURNING:
            total_steps = self.config.return_steps
            current_step = self._return_step

        progress = current_step / total_steps if total_steps > 0 else 0.0

        return {
            "phase": self.phase.name,
            "is_pick": self._is_pick,
            "current_step": current_step,
            "total_steps": total_steps,
            "progress": progress,
        }
