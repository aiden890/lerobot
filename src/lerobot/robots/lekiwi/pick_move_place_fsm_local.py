"""Local inference FSM controller for lekiwi pick-move-place routines.

This version runs policy inference locally on MacBook (MPS/CPU) instead of
communicating with a remote policy server via gRPC.
"""

from collections import deque
import json
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
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, map_robot_keys_to_lerobot_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots.robot import Robot
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait
from .config_lekiwi import LeKiwiClientConfig
from .lekiwi_client import LeKiwiClient
from .follow_line import (
    VerticalLineAligner,
    FollowState,
    FORWARD_SPEED,
    TURN_SPEED,
    TURN_SPEED_ON_INITIAL,
    TURN_SPEED_ON_RESUME,
    TURN_ANGLE_ON_RESUME,
    LATERAL_SPEED_ON_DIAGONAL,
    LATERAL_SPEED_ON_RESUME,
)


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

    state: PickPlaceState = PickPlaceState.IDLE
    _pending_command: PickPlaceCommand | None = None
    _force_next_state_event: threading.Event = field(default_factory=threading.Event)

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

    # State handlers -----------------------------------------------------
    def _handle_idle(self) -> None:
        if self._pending_command == PickPlaceCommand.START:
            self._pending_command = None
            self.controller.start_pick()
            self.state = PickPlaceState.PICK
        elif self._pending_command == PickPlaceCommand.STOP:
            self._pending_command = None
        elif self._check_force_next_state():
            logging.info("Force starting PICK from IDLE state")
            self.controller.start_pick()
            self.state = PickPlaceState.PICK

    def _handle_pick(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        if self._check_force_next_state():
            logging.info("Force skipping PICK state")
            self.controller.follow_path(self.config.pick_to_place_path)
            self.state = PickPlaceState.GOTO_PLACE

    def _handle_goto_place(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        if self._check_force_next_state():
            logging.info("Force skipping GOTO_PLACE state")
            self.controller.start_place()
            self.state = PickPlaceState.PLACE

    def _handle_place(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        if self._check_force_next_state():
            logging.info("Force skipping PLACE state")
            self.controller.follow_path(self.config.place_to_pick_path)
            self.state = PickPlaceState.GOTO_PICK

    def _handle_goto_pick(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        if self._check_force_next_state():
            logging.info("Force skipping GOTO_PICK state")
            self.state = PickPlaceState.IDLE

    def _abort(self) -> None:
        self._pending_command = None
        self.controller.cancel()
        self.state = PickPlaceState.IDLE


@dataclass
class PresetDatasetConfig:
    repo_id: str
    episode_index: int = 0


@dataclass
class LocalExecutorConfig:
    """Configuration for local inference controller."""

    policy_config: RemotePolicyConfig
    environment_dt: float = 1 / 30
    logger_name: str = "lekiwi_local_executor"
    task: str = "pick the green object and place it on the white"
    preset_datasets: dict[str, PresetDatasetConfig] = field(default_factory=dict)
    gripper_open_threshold: float = 70.0
    gripper_open_duration_s: float = 1.5
    gripper_stable_range: float = 5.0
    gripper_closed_threshold: float = 30.0
    gripper_closed_duration_s: float = 1.5
    queue_refill_threshold: float = 0.2  # Refill when queue is below this ratio (0.2 = 20%)


class LocalInferencePickPlaceController(PickPlaceController):
    """Controls both arm and base behaviour using local policy inference."""

    def __init__(self, config: LocalExecutorConfig, robot: Robot):
        self.config = config
        self._robot = robot
        self._logger = logging.getLogger(config.logger_name)
        self._current_phase: str | None = None
        self._phase_lock = threading.Lock()
        self._path_done_event = threading.Event()
        self._path_done_event.set()
        self._action_keys = list(robot.action_features.keys())
        self._task_instruction = config.task

        # Preset actions
        self._preset_actions: dict[str, list[dict[str, float]]] = {}
        self._preset_indices: dict[str, int] = {}
        if config.preset_datasets:
            self._load_preset_datasets(config.preset_datasets)

        # Gripper tracking
        self._gripper_open_threshold = config.gripper_open_threshold
        self._gripper_open_duration_s = config.gripper_open_duration_s
        self._gripper_stable_range = config.gripper_stable_range
        self._gripper_open_since: float | None = None
        self._gripper_stable_value: float | None = None
        self._gripper_closed_threshold = config.gripper_closed_threshold
        self._gripper_closed_duration_s = config.gripper_closed_duration_s
        self._gripper_closed_since: float | None = None

        # Line following
        self._line_aligner: VerticalLineAligner | None = None
        self._line_follow_target: FollowState | None = None
        self._is_line_following: bool = False
        self._line_follow_none_count: int = 0  # For throttling warning logs

        # Local inference setup
        self._device = self._select_device(config.policy_config.device)
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._action_queue: deque = deque()
        self._actions_per_chunk = config.policy_config.actions_per_chunk
        self._policy_loaded = False

        # Current observation for inference
        self._current_observation: dict[str, Any] | None = None
        self._observation_timestamp: float = 0.0  # When observation was captured

        # Async inference threading
        self._action_queue_lock = threading.Lock()
        self._obs_lock = threading.Lock()
        self._inference_event = threading.Event()  # Signal to run inference
        self._inference_thread: threading.Thread | None = None
        self._inference_shutdown = threading.Event()  # Signal to stop inference thread
        self._inference_running = False  # Flag to track if inference is currently running

    def _select_device(self, requested_device: str) -> str:
        """Select the best available device."""
        if requested_device == "mps":
            if torch.backends.mps.is_available():
                self._logger.info("Using MPS (Metal Performance Shaders) device")
                return "mps"
            else:
                self._logger.warning("MPS not available, falling back to CPU")
                return "cpu"
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                self._logger.info("Using CUDA device")
                return "cuda"
            else:
                self._logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
        else:
            self._logger.info("Using CPU device")
            return "cpu"

    def _load_policy(self) -> None:
        """Load policy and create preprocessor/postprocessor."""
        if self._policy_loaded:
            return

        self._logger.info("Loading policy from %s", self.config.policy_config.pretrained_name_or_path)
        start_time = time.perf_counter()

        # Get policy class and load
        policy_class = get_policy_class(self.config.policy_config.policy_type)
        self._policy = policy_class.from_pretrained(
            self.config.policy_config.pretrained_name_or_path
        )
        self._policy.to(self._device)
        self._policy.eval()

        # Create preprocessor and postprocessor with device override
        device_override = {"device": str(self._device)}
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=self._policy.config,
            pretrained_path=self.config.policy_config.pretrained_name_or_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )

        elapsed = time.perf_counter() - start_time
        self._logger.info("Policy loaded in %.2fs on device: %s", elapsed, self._device)
        self._policy_loaded = True

    def _prepare_observation_for_inference(self, observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Convert robot observation to format expected by policy."""
        batch = {}

        # Get lerobot features mapping
        if not self.config.policy_config.lerobot_features:
            self.config.policy_config.lerobot_features = map_robot_keys_to_lerobot_features(self._robot)

        lerobot_features = self.config.policy_config.lerobot_features

        # Process state observations
        state_keys = [k for k in observation.keys() if not isinstance(observation[k], np.ndarray) or observation[k].ndim < 3]
        state_values = []
        for key in self._action_keys:
            if key in observation:
                state_values.append(float(observation[key]))
            else:
                state_values.append(0.0)

        # Add state tensor
        state_tensor = torch.tensor(state_values, dtype=torch.float32, device=self._device)
        batch["observation.state"] = state_tensor.unsqueeze(0)  # Add batch dim

        # Process image observations
        for cam_name, cam_cfg in self._robot.config.cameras.items():
            if cam_name in observation:
                image = observation[cam_name]
                if isinstance(image, np.ndarray):
                    # Convert BGR to RGB if needed, normalize to [0, 1]
                    if image.shape[-1] == 3:
                        image = image[..., ::-1].copy()  # BGR to RGB
                    image_tensor = torch.from_numpy(image).float() / 255.0
                    # Reshape to (C, H, W)
                    if image_tensor.ndim == 3:
                        image_tensor = image_tensor.permute(2, 0, 1)
                    batch[f"observation.images.{cam_name}"] = image_tensor.unsqueeze(0).to(self._device)

        # Add task instruction if needed
        if self._task_instruction:
            batch["task"] = self._task_instruction

        return batch

    def _run_inference_internal(self, observation: dict[str, Any], obs_timestamp: float) -> list[TimedAction]:
        """Run local inference and return list of TimedActions.

        This is the actual inference logic, called from the background thread.
        """
        if not self._policy_loaded:
            self._load_policy()

        try:
            t_start = time.perf_counter()

            # Prepare observation
            batch = self._prepare_observation_for_inference(observation)

            # Preprocess
            if self._preprocessor:
                batch = self._preprocessor(batch)

            # Run inference
            with torch.inference_mode():
                action_chunk = self._policy.predict_action_chunk(batch)

            # Ensure correct shape: (batch, chunk_size, action_dim)
            if action_chunk.ndim == 2:
                action_chunk = action_chunk.unsqueeze(0)

            # Limit to configured actions per chunk
            action_chunk = action_chunk[:, :self._actions_per_chunk, :]

            # Create TimedActions with proper timestamps
            # Each action[i] should be executed at obs_timestamp + i * dt
            dt = self.config.environment_dt  # Time between actions (e.g., 1/30 = 0.0333s)

            timed_actions = []
            for i in range(action_chunk.shape[1]):
                action = action_chunk[0, i, :]  # Remove batch dim
                if self._postprocessor:
                    action = self._postprocessor(action)

                # Calculate expected execution time for this action
                expected_timestamp = obs_timestamp + i * dt
                timed_action = TimedAction(
                    timestamp=expected_timestamp,
                    timestep=i,
                    action=action
                )
                timed_actions.append(timed_action)

            t_elapsed = (time.perf_counter() - t_start) * 1000
            self._logger.info("[AsyncInference] Generated %d actions in %.1fms", len(timed_actions), t_elapsed)
            return timed_actions

        except Exception as e:
            self._logger.error("Inference failed: %s", e)
            import traceback
            traceback.print_exc()
            return []

    def _inference_loop(self) -> None:
        """Background thread loop for async inference."""
        self._logger.info("[AsyncInference] Inference thread started")

        while not self._inference_shutdown.is_set():
            # Wait for signal to run inference (with timeout to check shutdown)
            signaled = self._inference_event.wait(timeout=0.1)

            if not signaled:
                continue

            if self._inference_shutdown.is_set():
                break

            self._inference_event.clear()
            self._inference_running = True

            try:
                # Copy observation with lock
                with self._obs_lock:
                    if self._current_observation is None:
                        self._logger.warning("[AsyncInference] No observation available")
                        self._inference_running = False
                        continue
                    obs_copy = {k: v.copy() if hasattr(v, 'copy') else v
                               for k, v in self._current_observation.items()}
                    obs_timestamp = self._observation_timestamp

                # Run inference (SLOW - but doesn't block main loop)
                timed_actions = self._run_inference_internal(obs_copy, obs_timestamp)

                if timed_actions:
                    # Clear old actions and add new ones with lock
                    with self._action_queue_lock:
                        self._action_queue.clear()
                        self._action_queue.extend(timed_actions)
                        self._logger.debug("[AsyncInference] Queue updated: %d actions", len(self._action_queue))

            except Exception as e:
                self._logger.error("[AsyncInference] Error in inference loop: %s", e)
            finally:
                self._inference_running = False

        self._logger.info("[AsyncInference] Inference thread stopped")

    def start_pick(self) -> None:
        """Start pick phase with policy-guided actions."""
        self._load_policy()
        self._path_done_event.set()
        self._set_phase("pick")
        self._gripper_open_since = None
        self._gripper_stable_value = None
        with self._action_queue_lock:
            self._action_queue.clear()
        self._logger.info("Pick phase started (local inference)")

    def pick_complete(self) -> bool:
        """Check if gripper has been open and stable for required duration."""
        if self._gripper_open_since is None or self._gripper_stable_value is None:
            return False

        elapsed = time.time() - self._gripper_open_since
        if elapsed >= self._gripper_open_duration_s:
            self._logger.info(
                "Pick complete: gripper held stable at %.1f degrees for %.2fs",
                self._gripper_stable_value,
                elapsed,
            )
            return True
        return False

    def start_place(self) -> None:
        """Start place phase with policy-guided actions."""
        self._load_policy()
        self._path_done_event.set()
        self._set_phase("place")
        self._gripper_closed_since = None
        with self._action_queue_lock:
            self._action_queue.clear()
        self._logger.info("Place phase started (local inference)")

    def place_complete(self) -> bool:
        """Check if gripper has been closed for required duration."""
        if self._gripper_closed_since is None:
            return False

        elapsed = time.time() - self._gripper_closed_since
        if elapsed >= self._gripper_closed_duration_s:
            self._logger.info("Place complete: gripper held closed for %.2fs", elapsed)
            return True
        return False

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

        # Fallback to preset actions
        if path_id not in self._preset_actions:
            self._logger.warning("No preset actions registered for path '%s'", path_id)
            self._path_done_event.set()
            return

        self._logger.info("Starting path '%s'", path_id)
        self._path_done_event.clear()
        self._set_phase(path_id)

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
        with self._action_queue_lock:
            self._action_queue.clear()

    def build_action(self) -> dict[str, float] | None:
        """Non-blocking action builder. Signals inference thread when needed."""
        phase = self._current_phase
        if not phase:
            return None

        if phase == "pick" or phase == "place":
            # Check queue status with lock
            with self._action_queue_lock:
                queue_len = len(self._action_queue)
                queue_ratio = queue_len / self._actions_per_chunk if self._actions_per_chunk > 0 else 0

            # Signal inference thread if queue is low (non-blocking)
            if queue_ratio < self.config.queue_refill_threshold and not self._inference_running:
                self._inference_event.set()  # Signal background thread to run inference

            # Get action from queue (fast, non-blocking)
            with self._action_queue_lock:
                if len(self._action_queue) > 0:
                    current_time = time.perf_counter()
                    skipped_count = 0

                    # Skip actions whose expected execution time has passed
                    while len(self._action_queue) > 0:
                        timed_action = self._action_queue[0]  # Peek at front
                        if timed_action.get_timestamp() <= current_time:
                            self._action_queue.popleft()
                            skipped_count += 1
                        else:
                            break

                    if skipped_count > 0:
                        self._logger.debug("[TimedAction] Skipped %d past actions", skipped_count)

                    # Get the appropriate action
                    if len(self._action_queue) > 0:
                        timed_action = self._action_queue.popleft()
                        action = self._tensor_to_action_dict(timed_action.get_action())

                        # Log timing info
                        action_delay = (current_time - timed_action.get_timestamp()) * 1000
                        if skipped_count > 0:
                            self._logger.info("[Timing] action_delay: %.1fms, skipped: %d, queue: %d",
                                             action_delay, skipped_count, len(self._action_queue))
                        return self._filter_pick_action(action)
            return None

        # Line following phases
        if self._is_line_following and self._line_aligner:
            action = self._build_line_follow_action()
            if action is None:
                self._line_follow_none_count += 1
                if self._line_follow_none_count % 30 == 1:  # Log every ~1 second at 30Hz
                    self._logger.warning("Line follow action is None! (count: %d)", self._line_follow_none_count)
            else:
                self._line_follow_none_count = 0  # Reset on success
            return action

        # Fallback to preset actions
        preset = self._get_preset_action(phase)
        if preset is None:
            self._path_done_event.set()
        return preset

    def update_observation(self, observation: dict[str, Any]) -> None:
        """Update current observation for inference (thread-safe)."""
        with self._obs_lock:
            self._current_observation = observation
            # Use receive timestamp from client if available, otherwise use current time
            self._observation_timestamp = observation.get("_receive_timestamp", time.perf_counter())
        self._update_gripper_state(observation)

    def start(self) -> None:
        """Start controller and inference thread."""
        self._load_policy()

        # Start inference thread
        if self._inference_thread is None or not self._inference_thread.is_alive():
            self._inference_shutdown.clear()
            self._inference_thread = threading.Thread(
                target=self._inference_loop,
                name="InferenceThread",
                daemon=True
            )
            self._inference_thread.start()
            self._logger.info("Started inference thread")

    def stop(self) -> None:
        """Stop controller and inference thread."""
        # Signal inference thread to stop
        self._inference_shutdown.set()
        self._inference_event.set()  # Wake up thread if waiting

        # Wait for thread to finish
        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)
            if self._inference_thread.is_alive():
                self._logger.warning("Inference thread did not stop gracefully")
            else:
                self._logger.info("Inference thread stopped")
            self._inference_thread = None

        with self._action_queue_lock:
            self._action_queue.clear()
        with self._obs_lock:
            self._current_observation = None

    def _tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        if isinstance(action_tensor, torch.Tensor):
            flat = action_tensor.detach().cpu().view(-1).tolist()
        else:
            flat = list(action_tensor)
        if len(flat) < len(self._action_keys):
            flat.extend([0.0] * (len(self._action_keys) - len(flat)))
        return {key: float(flat[i]) for i, key in enumerate(self._action_keys)}

    def _filter_pick_action(self, action: dict[str, float]) -> dict[str, float]:
        filtered = dict(action)
        for key in ("x.vel", "y.vel", "theta.vel"):
            if key in filtered:
                filtered[key] = 0.0
        return self._normalize_action_dict(filtered)

    def _set_phase(self, phase: str | None) -> None:
        with self._phase_lock:
            self._current_phase = phase
            if phase and phase in self._preset_actions:
                self._preset_indices[phase] = 0
            if phase != "pick":
                self._gripper_open_since = None
                self._gripper_stable_value = None
            if phase != "place":
                self._gripper_closed_since = None

    def _normalize_action_dict(self, action: dict[str, float]) -> dict[str, float]:
        return {key: float(action.get(key, 0.0)) for key in self._action_keys}

    def _get_preset_action(self, phase: str) -> dict[str, float] | None:
        sequence = self._preset_actions.get(phase)
        if not sequence:
            return None

        idx = self._preset_indices.get(phase, 0)
        if idx >= len(sequence):
            return None

        action = self._normalize_action_dict(sequence[idx])
        self._preset_indices[phase] = idx + 1
        if self._preset_indices[phase] >= len(sequence):
            self._path_done_event.set()
        return action

    def _load_preset_datasets(self, presets: dict[str, PresetDatasetConfig]) -> None:
        for phase, preset_cfg in presets.items():
            try:
                sequence = self._load_actions_from_dataset(preset_cfg)
            except Exception as err:
                self._logger.error("Failed to load preset dataset for '%s': %s", phase, err)
                continue

            if not sequence:
                self._logger.warning("Preset dataset for phase '%s' produced no actions", phase)
                continue

            self._preset_actions[phase] = sequence
            self._preset_indices[phase] = 0

    def _load_actions_from_dataset(self, preset_cfg: PresetDatasetConfig) -> list[dict[str, float]]:
        dataset = LeRobotDataset(preset_cfg.repo_id, episodes=[preset_cfg.episode_index])
        episode_frames = dataset.hf_dataset.filter(
            lambda row, ep=preset_cfg.episode_index: row["episode_index"] == ep
        )
        actions_only = episode_frames.select_columns(ACTION)
        action_names = dataset.features[ACTION]["names"]

        sequence: list[dict[str, float]] = []
        for idx in range(len(actions_only)):
            raw = actions_only[idx][ACTION]
            action_dict = {name: float(raw[i]) for i, name in enumerate(action_names)}
            sequence.append(self._normalize_action_dict(action_dict))
        return sequence

    def _update_gripper_state(self, observation: dict[str, Any]) -> None:
        value = observation.get("arm_gripper.pos")
        if value is None:
            self._gripper_open_since = None
            self._gripper_stable_value = None
            self._gripper_closed_since = None
            return

        # Track gripper open and stable state (for pick phase)
        if value >= self._gripper_open_threshold:
            if self._gripper_stable_value is None:
                self._gripper_open_since = time.time()
                self._gripper_stable_value = value
            elif abs(value - self._gripper_stable_value) <= self._gripper_stable_range:
                pass
            else:
                self._gripper_open_since = time.time()
                self._gripper_stable_value = value
        else:
            self._gripper_open_since = None
            self._gripper_stable_value = None

        # Track gripper closed state (for place phase)
        if value <= self._gripper_closed_threshold:
            if self._gripper_closed_since is None:
                self._gripper_closed_since = time.time()
        else:
            self._gripper_closed_since = None

    # =========================================================================
    # Line Following Methods
    # =========================================================================

    def _start_line_following(self, target_state: FollowState, needs_initial_turn: bool = False, needs_lateral_turn: bool = False) -> None:
        """Initialize line following towards the target state."""
        self._logger.info("_start_line_following called with target=%s, needs_initial_turn=%s, needs_lateral_turn=%s",
                         target_state.name, needs_initial_turn, needs_lateral_turn)

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
        self._logger.info("Line following initialized: _is_line_following=%s, aligner_state=%s, phase=%s",
                         self._is_line_following, self._line_aligner.state.name, self._current_phase)

    def _build_line_follow_action(self) -> dict[str, float] | None:
        """Build action for line following based on camera input."""
        if not self._line_aligner:
            return None

        dt = self.config.environment_dt

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

    def _compute_line_action(self, vertical_polylines: list, image_width: int, image_height: int) -> dict[str, float] | None:
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
class LocalPickPlaceLoopConfig:
    controller: LocalExecutorConfig
    robot: LeKiwiClientConfig = field(default_factory=LeKiwiClientConfig)
    fsm: PickPlaceConfig = field(default_factory=PickPlaceConfig)
    target_hz: float = 30.0
    auto_start: bool = True
    enable_keyboard_control: bool = True


def _keyboard_listener(fsm: LeKiwiPickPlaceFSM, shutdown_event: threading.Event) -> None:
    """Listen for keyboard input to force state transitions."""
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        logging.info("Keyboard control enabled. Press 'n' to skip to next state, 'q' to quit.")

        while not shutdown_event.is_set():
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == 'n':
                    logging.info("User requested force next state (current: %s)", fsm.state.name)
                    fsm.force_next_state()
                elif key == 'q':
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
    controller: LocalInferencePickPlaceController,
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

    # Performance profiling
    profile_window = 30
    fsm_times = []
    build_action_times = []
    send_action_times = []
    get_obs_times = []

    while not shutdown_event.is_set():
        loop_start = time.perf_counter()

        # FSM step
        t0 = time.perf_counter()
        fsm.step()
        fsm_times.append(time.perf_counter() - t0)

        # Get observation first (for local inference)
        t0 = time.perf_counter()
        observation = robot.get_observation()
        get_obs_times.append(time.perf_counter() - t0)

        # Update controller with observation
        controller.update_observation(observation)

        # Build action
        t0 = time.perf_counter()
        action = controller.build_action()
        build_action_times.append(time.perf_counter() - t0)

        # Send action
        if action:
            t0 = time.perf_counter()
            robot.send_action(action)
            send_action_times.append(time.perf_counter() - t0)

        timestep += 1

        elapsed = time.perf_counter() - loop_start
        busy_wait(max(0.0, period - elapsed))

        # Track FPS
        total_time = time.perf_counter() - loop_start
        loop_times.append(total_time)
        if len(loop_times) > fps_window_size:
            loop_times.pop(0)

        # Trim profiling lists
        for lst in [fsm_times, build_action_times, send_action_times, get_obs_times]:
            while len(lst) > profile_window:
                lst.pop(0)

        # Print FPS and profiling info
        current_time = time.perf_counter()
        if current_time - last_print_time >= print_interval:
            avg_loop_time = sum(loop_times) / len(loop_times)
            current_fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

            avg_fsm = sum(fsm_times) / len(fsm_times) * 1000 if fsm_times else 0
            avg_build = sum(build_action_times) / len(build_action_times) * 1000 if build_action_times else 0
            avg_send_act = sum(send_action_times) / len(send_action_times) * 1000 if send_action_times else 0
            avg_get_obs = sum(get_obs_times) / len(get_obs_times) * 1000 if get_obs_times else 0

            print(
                f"\rFPS: {current_fps:.1f}/{target_hz:.0f} | "
                f"FSM: {avg_fsm:.1f}ms | Build: {avg_build:.1f}ms | SendAct: {avg_send_act:.1f}ms | "
                f"GetObs: {avg_get_obs:.1f}ms | Step: {timestep}",
                end='', flush=True
            )
            last_print_time = current_time


@draccus.wrap()
def pick_place_control_loop(cfg: LocalPickPlaceLoopConfig) -> None:
    logging.info("Starting pick-place control loop (LOCAL INFERENCE)")

    robot = LeKiwiClient(cfg.robot)
    robot.connect()
    controller = LocalInferencePickPlaceController(cfg.controller, robot)
    fsm = LeKiwiPickPlaceFSM(controller=controller, config=cfg.fsm)

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
    log_file = log_dir / f"fsm_local_{timestamp}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ]
    )
    logging.info(f"Log file: {log_file}")
    logging.info("Running LOCAL INFERENCE mode (no policy server)")
    pick_place_control_loop()


if __name__ == "__main__":
    main()
