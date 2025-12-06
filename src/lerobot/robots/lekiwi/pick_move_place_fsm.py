"""Skeleton FSM controller for lekiwi pick-move-place routines."""

from collections import deque
import json
import logging
import pickle
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
import grpc
import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation, map_robot_keys_to_lerobot_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.robot import Robot
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
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

    def send_observation(self, observation: dict[str, Any], timestep: int, must_go: bool) -> None:
        ...

    def should_send_observation(self) -> bool:
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
        # Manual transition with 'n' key (thread-safe check)
        elif self._check_force_next_state():
            logging.info("Force starting PICK from IDLE state")
            self.controller.start_pick()
            self.state = PickPlaceState.PICK

    def _handle_pick(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        # Manual transition with 'n' key (thread-safe check)
        if self._check_force_next_state():
            logging.info("Force skipping PICK state")
            self.controller.follow_path(self.config.pick_to_place_path)
            self.state = PickPlaceState.GOTO_PLACE

    def _handle_goto_place(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        # Manual transition with 'n' key (thread-safe check)
        if self._check_force_next_state():
            logging.info("Force skipping GOTO_PLACE state")
            self.controller.start_place()
            self.state = PickPlaceState.PLACE

    def _handle_place(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        # Manual transition with 'n' key (thread-safe check)
        if self._check_force_next_state():
            logging.info("Force skipping PLACE state")
            self.controller.follow_path(self.config.place_to_pick_path)
            self.state = PickPlaceState.GOTO_PICK

    def _handle_goto_pick(self) -> None:
        if self._pending_command == PickPlaceCommand.STOP:
            self._abort()
            return
        # Manual transition with 'n' key (thread-safe check)
        if self._check_force_next_state():
            logging.info("Force skipping GOTO_PICK state")
            self.state = PickPlaceState.IDLE

    # Helpers ------------------------------------------------------------
    def _abort(self) -> None:
        self._pending_command = None
        self.controller.cancel()
        self.state = PickPlaceState.IDLE


@dataclass
class PresetDatasetConfig:
    repo_id: str
    episode_index: int = 0


@dataclass
class PolicyServerExecutorConfig:
    """Configuration required to talk to the remote policy server."""

    server_address: str
    policy_config: RemotePolicyConfig
    environment_dt: float = 1 / 30
    logger_name: str = "lekiwi_vla_executor"
    task: str = "pick the green object and place it on the white"
    preset_datasets: dict[str, PresetDatasetConfig] = field(default_factory=dict)
    gripper_open_threshold: float = 70.0
    gripper_open_duration_s: float = 1.5
    gripper_stable_range: float = 5.0  # Allowable variation in gripper position (degrees)
    gripper_closed_threshold: float = 30.0
    gripper_closed_duration_s: float = 1.5
    chunk_size_threshold: float = 0.5  # Send observation when queue_size / actions_per_chunk <= threshold
    aggregate_fn_name: str = "latest_only"  # Function to aggregate overlapping timestep actions


class PolicyServerPickPlaceController(PickPlaceController):
    """Controls both arm and base behaviour around the policy server."""

    def __init__(self, config: PolicyServerExecutorConfig, robot: Robot):
        self.config = config
        self._robot = robot  # Store robot reference for line following
        self._channel: grpc.Channel | None = None
        self._stub: services_pb2_grpc.AsyncInferenceStub | None = None
        self._connected = False
        self._logger = logging.getLogger(config.logger_name)
        self._current_phase: str | None = None
        self._action_thread: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._worker_started = False
        self._pending_actions: deque[TimedAction] = deque()
        self._pending_actions_lock = threading.Lock()
        self._phase_lock = threading.Lock()
        self._path_done_event = threading.Event()
        self._path_done_event.set()
        self._action_keys = list(robot.action_features.keys())
        self._task_instruction = config.task
        self._preset_actions: dict[str, list[dict[str, float]]] = {}
        self._preset_indices: dict[str, int] = {}
        if config.preset_datasets:
            self._load_preset_datasets(config.preset_datasets)
        self._gripper_open_threshold = config.gripper_open_threshold
        self._gripper_open_duration_s = config.gripper_open_duration_s
        self._gripper_stable_range = config.gripper_stable_range
        self._gripper_open_since: float | None = None
        self._gripper_stable_value: float | None = None  # Track stable gripper position
        self._gripper_closed_threshold = config.gripper_closed_threshold
        self._gripper_closed_duration_s = config.gripper_closed_duration_s
        self._gripper_closed_since: float | None = None
        self._last_executed_timestep = -1
        self._action_chunk_size = config.policy_config.actions_per_chunk
        self._chunk_size_threshold = config.chunk_size_threshold
        self._aggregate_fn = get_aggregate_function(config.aggregate_fn_name)
        self._ensure_policy_features(robot)

        # Line following variables
        self._line_aligner: VerticalLineAligner | None = None
        self._line_follow_target: FollowState | None = None  # ARRIVED or INITIAL
        self._is_line_following: bool = False
        
    @property
    def stub(self) -> services_pb2_grpc.AsyncInferenceStub:
        if not self._stub:
            raise RuntimeError("Policy server stub not initialized. Call connect() first.")
        return self._stub

    def connect(self) -> None:
        if self._connected:
            return

        self._logger.info("Connecting to policy server at %s", self.config.server_address)
        channel = grpc.insecure_channel(
            self.config.server_address,
            grpc_channel_options(initial_backoff=f"{self.config.environment_dt:.4f}s"),
        )
        stub = services_pb2_grpc.AsyncInferenceStub(channel)

        try:
            stub.Ready(services_pb2.Empty())
            policy_bytes = pickle.dumps(self.config.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_bytes)
            stub.SendPolicyInstructions(policy_setup)
        except grpc.RpcError as err:  # pragma: no cover - network failures are environment specific
            channel.close()
            self._logger.error("Failed to initialize policy server connection: %s", err)
            raise RuntimeError("Policy server handshake failed") from err

        self._channel = channel
        self._stub = stub
        self._connected = True
        self._logger.info("Policy server handshake completed")

    def disconnect(self) -> None:
        self._shutdown.set()
        self._set_phase(None)
        if self._action_thread and self._action_thread.is_alive():
            self._action_thread.join(timeout=1)
        self._worker_started = False
        if self._channel:
            self._channel.close()
        self._channel = None
        self._stub = None
        self._connected = False

    def start_pick(self) -> None:
        """Start pick phase with policy-guided actions."""

        self.connect()
        self._ensure_worker_started()

        self._path_done_event.set()
        self._set_phase("pick")
        self._gripper_open_since = None
        self._gripper_stable_value = None
        self._logger.info("Pick phase started")

    def pick_complete(self) -> bool:
        """Check if gripper has been open and stable at a consistent angle for required duration."""
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
        self.connect()
        self._ensure_worker_started()

        self._path_done_event.set()
        self._set_phase("place")
        self._gripper_closed_since = None
        self._logger.info("Place phase started")

    def place_complete(self) -> bool:
        """Check if gripper has been closed below threshold for required duration."""
        if self._gripper_closed_since is None:
            return False

        elapsed = time.time() - self._gripper_closed_since
        if elapsed >= self._gripper_closed_duration_s:
            self._logger.info("Place complete: gripper held closed for %.2fs", elapsed)
            return True
        return False

    def follow_path(self, path_id: str) -> None:
        """Start executing a named base path using line following."""

        self.connect()
        self._ensure_worker_started()

        # Use line following for pick_to_place and place_to_pick paths
        if path_id == "pick_to_place":
            # pick_to_place: 먼저 130도 회전 후 line following 시작
            self._start_line_following(target_state=FollowState.ARRIVED, needs_initial_turn=True)
            self._logger.info("Starting line following to ARRIVED (pick_to_place) with initial 130° turn")
            return
        elif path_id == "place_to_pick":
            # place_to_pick: 왼쪽 이동 → 130도 회전 후 line following 시작
            self._start_line_following(target_state=FollowState.INITIAL, needs_lateral_turn=True)
            self._logger.info("Starting line following to INITIAL (place_to_pick) with lateral left + 130° turn")
            return

        # Fallback to preset actions for other paths
        if path_id not in self._preset_actions:
            self._logger.warning("No preset actions registered for path '%s'", path_id)
            self._path_done_event.set()
            return

        self._logger.info("Starting path '%s'", path_id)
        self._path_done_event.clear()
        self._set_phase(path_id)

    def motion_done(self) -> bool:
        # Check line following completion
        if self._is_line_following and self._line_aligner:
            if self._line_aligner.state == self._line_follow_target:
                self._is_line_following = False
                self._path_done_event.set()
                self._logger.info("Line following complete: reached %s", self._line_follow_target.name)
                return True
            return False
        return self._path_done_event.is_set()

    def cancel(self) -> None:
        """Abort current activity and stop the robot safely."""

        self._logger.info("Cancelling current phase")
        self._set_phase(None)
        self._path_done_event.set()
        self._clear_pending_actions()

    def build_action(self) -> dict[str, float] | None:
        phase = self._current_phase
        if not phase:
            return None

        if phase == "pick" or phase == "place":
            timed_action = self._next_timed_action()
            if not timed_action:
                return None
            action = self._tensor_to_action_dict(timed_action.get_action())
            return self._filter_pick_action(action)

        # Line following phases
        if self._is_line_following and self._line_aligner:
            action = self._build_line_follow_action()
            if action is None:
                self._logger.warning("Line follow action is None!")
            return action
        elif phase and phase.startswith("line_follow"):
            self._logger.warning("Line follow phase but not executing: _is_line_following=%s, _line_aligner=%s",
                                self._is_line_following, self._line_aligner is not None)

        # Fallback to preset actions
        preset = self._get_preset_action(phase)
        if preset is None:
            self._path_done_event.set()
        return preset

    def send_observation(self, observation: dict[str, Any], timestep: int, must_go: bool) -> None:
        if not self._stub:
            raise RuntimeError("Controller is not connected to the policy server")

        t_prep_start = time.perf_counter()
        payload = dict(observation)
        if self._task_instruction:
            payload.setdefault("task", self._task_instruction)

        self._update_gripper_state(payload)

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=timestep,
            observation=payload,
            must_go=must_go,
        )
        t_prep = (time.perf_counter() - t_prep_start) * 1000

        try:
            t_serial_start = time.perf_counter()
            serialized = pickle.dumps(timed_observation)
            t_serial = (time.perf_counter() - t_serial_start) * 1000

            t_chunk_start = time.perf_counter()
            iterator = send_bytes_in_chunks(
                serialized,
                services_pb2.Observation,
                log_prefix="[lekiwi] Observation",
                silent=True,
            )
            t_chunk = (time.perf_counter() - t_chunk_start) * 1000

            t_grpc_start = time.perf_counter()
            self._stub.SendObservations(iterator)
            t_grpc = (time.perf_counter() - t_grpc_start) * 1000

            if timestep % 15 == 0:  # Log every 15 steps
                print(
                    f"SendObs breakdown: Prep={t_prep:.1f}ms Serialize={t_serial:.1f}ms "
                    f"Chunk={t_chunk:.1f}ms gRPC={t_grpc:.1f}ms"
                )
        except grpc.RpcError as err:  # pragma: no cover - network-related failures
            self._logger.error("Failed to send observation #%s: %s", timestep, err)

    def should_send_observation(self) -> bool:
        """Check if we should send observation based on action queue size."""
        with self._pending_actions_lock:
            queue_size = len(self._pending_actions)
        return queue_size / self._action_chunk_size <= self._chunk_size_threshold

    def start(self) -> None:
        self.connect()
        self._ensure_worker_started()

    def stop(self) -> None:
        self.disconnect()

    def _ensure_worker_started(self) -> None:
        if self._worker_started:
            return

        self._shutdown.clear()
        self._action_thread = threading.Thread(target=self._action_receiver_loop, name="lekiwi_actions", daemon=True)
        self._action_thread.start()
        self._worker_started = True

    def _action_receiver_loop(self) -> None:
        self._logger.info("Action receiver loop started")
        while not self._shutdown.is_set():
            if not self._connected or not self._stub:
                time.sleep(0.1)
                continue

            try:
                actions_chunk = self._stub.GetActions(services_pb2.Empty())
            except grpc.RpcError as err:  # pragma: no cover - network-related failures
                self._logger.error("Error while receiving actions: %s", err)
                time.sleep(0.2)
                continue

            if not actions_chunk.data:
                continue

            try:
                timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)
            except Exception as err:  # pragma: no cover - corrupt payloads
                self._logger.error("Failed to deserialize actions: %s", err)
                continue

            # Aggregate actions based on timestep (similar to robot_client.py)
            with self._pending_actions_lock:
                # Convert current queue to timestep-based dict
                current_actions = {a.timestep: a for a in self._pending_actions}

                # Merge incoming actions with aggregation
                for action in timed_actions:
                    # Skip actions that have already been executed
                    if action.timestep <= self._last_executed_timestep:
                        continue

                    # If timestep exists, aggregate old and new actions
                    if action.timestep in current_actions:
                        old_action = current_actions[action.timestep]
                        aggregated_action_tensor = self._aggregate_fn(
                            old_action.action, action.action
                        )
                        current_actions[action.timestep] = TimedAction(
                            timestamp=action.timestamp,
                            timestep=action.timestep,
                            action=aggregated_action_tensor,
                        )
                    else:
                        # New timestep, just add it
                        current_actions[action.timestep] = action

                # Rebuild queue sorted by timestep
                self._pending_actions.clear()
                for ts in sorted(current_actions.keys()):
                    self._pending_actions.append(current_actions[ts])

    def _next_timed_action(self) -> TimedAction | None:
        with self._pending_actions_lock:
            if not self._pending_actions:
                return None
            action = self._pending_actions.popleft()
            self._last_executed_timestep = action.timestep
            return action

    def _clear_pending_actions(self) -> None:
        with self._pending_actions_lock:
            self._pending_actions.clear()

    def _tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        flat = action_tensor.detach().cpu().view(-1).tolist()
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
            except Exception as err:  # pragma: no cover - dataset access varies by setup
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

    def _ensure_policy_features(self, robot: Robot) -> None:
        if self.config.policy_config.lerobot_features:
            return

        features = map_robot_keys_to_lerobot_features(robot)
        self.config.policy_config.lerobot_features = features

    def _update_gripper_state(self, observation: dict[str, Any]) -> None:
        value = observation.get("arm_gripper.pos")
        if value is None:
            self._gripper_open_since = None
            self._gripper_stable_value = None
            self._gripper_closed_since = None
            return

        # Track gripper open and stable state (for pick phase)
        if value >= self._gripper_open_threshold:
            # Check if gripper position is stable
            if self._gripper_stable_value is None:
                # First time reaching open threshold, start tracking
                self._gripper_open_since = time.time()
                self._gripper_stable_value = value
            elif abs(value - self._gripper_stable_value) <= self._gripper_stable_range:
                # Gripper is stable within acceptable range, keep timing
                pass
            else:
                # Gripper moved outside stable range, reset tracking
                self._gripper_open_since = time.time()
                self._gripper_stable_value = value
        else:
            # Gripper closed below threshold, reset open tracking
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
        """Initialize line following towards the target state (ARRIVED or INITIAL)."""
        self._logger.info("_start_line_following called with target=%s, needs_initial_turn=%s, needs_lateral_turn=%s",
                         target_state.name, needs_initial_turn, needs_lateral_turn)

        if self._line_aligner is None:
            self._line_aligner = VerticalLineAligner(self._robot)
            self._logger.info("Created new VerticalLineAligner")

        # Reset aligner state to start fresh
        if needs_initial_turn:
            # pick_to_place: 130도 회전 후 line following
            self._line_aligner.state = FollowState.INITIAL_TURN
            self._line_aligner.angle_turned = 0.0
            self._line_aligner.action_phase = "forward"
            self._logger.info("Starting with INITIAL_TURN state (130° turn before line following)")
        elif needs_lateral_turn:
            # place_to_pick: 왼쪽 이동 → 130도 회전 후 line following
            self._line_aligner.state = FollowState.ARRIVED
            self._line_aligner.action_phase = "lateral"
            self._line_aligner.lateral_distance_traveled = 0.0
            self._line_aligner.angle_turned = 0.0
            self._line_aligner.turn_direction = -1.0  # 회전 시 오른쪽
            self._logger.info("Starting with ARRIVED state (lateral left → 130° turn before line following)")
        else:
            self._line_aligner.state = FollowState.FOLLOW_VERTICAL
            self._line_aligner.action_phase = "forward"

        self._line_aligner.horizontal_detected = False
        self._line_aligner.diagonal_detected = False
        self._line_aligner.distance_traveled = 0.0
        if not needs_lateral_turn:
            self._line_aligner.lateral_distance_traveled = 0.0
            self._line_aligner.turn_direction = -1.0  # 오른쪽 회전

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

        # Update state machine (handles INITIAL_TURN, FOLLOW_VERTICAL, etc.)
        self._line_aligner.update_state(horizontal, diagonal, width, height, dt)

        # Check if target reached
        if self._line_aligner.state == self._line_follow_target:
            self._is_line_following = False
            self._path_done_event.set()
            self._logger.info("Line following complete: reached %s", self._line_follow_target.name)
            return None

        # Compute action based on aligner state
        return self._compute_line_action(vertical, width, height)

    def _compute_line_action(self, vertical_polylines: list, image_width: int, image_height: int) -> dict[str, float] | None:
        """Compute action based on the current line aligner state."""
        aligner = self._line_aligner
        if not aligner:
            return None

        action = None

        if aligner.state == FollowState.INITIAL_TURN:
            # 초기 회전 (130도) - line following 시작 전
            action = aligner.compute_turn_control(TURN_SPEED_ON_RESUME)

        elif aligner.state == FollowState.FOLLOW_VERTICAL:
            # PD controller for vertical line following
            if vertical_polylines:
                result = aligner.compute_pd_control(vertical_polylines, image_width, image_height)
                if result is not None:
                    action, _, _, _, _ = result

        elif aligner.state == FollowState.HORIZONTAL_ACTION:
            # Horizontal line action: forward or turn
            if aligner.action_phase == "forward":
                action = aligner.compute_forward_control()
            elif aligner.action_phase == "turn":
                action = aligner.compute_turn_control(TURN_SPEED)

        elif aligner.state == FollowState.DIAGONAL_ACTION:
            # Diagonal line action (going to ARRIVED): forward then lateral right
            if aligner.action_phase == "forward":
                action = aligner.compute_forward_control()
            elif aligner.action_phase == "lateral":
                action = aligner.compute_lateral_control(LATERAL_SPEED_ON_DIAGONAL, direction=-1.0)  # 오른쪽

        elif aligner.state == FollowState.INITIAL_ACTION:
            # Initial action (going to INITIAL) - forward only
            action = aligner.compute_forward_control()

        elif aligner.state == FollowState.ARRIVED:
            # Arrived state - robot should stop or resume with lateral left then turn
            if aligner.action_phase == "lateral":
                action = aligner.compute_lateral_control(LATERAL_SPEED_ON_RESUME, direction=1.0)  # 왼쪽
            elif aligner.action_phase == "turn":
                action = aligner.compute_turn_control(TURN_SPEED_ON_RESUME)
            else:
                action = None  # Stop

        elif aligner.state == FollowState.INITIAL:
            # Initial state - robot should stop or resume turning
            if aligner.action_phase == "turn":
                action = aligner.compute_turn_control(TURN_SPEED_ON_INITIAL)
            else:
                action = None  # Stop

        return action


@dataclass
class PickPlaceLoopConfig:
    controller: PolicyServerExecutorConfig
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
    controller: PickPlaceController,
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
    print_interval = 0.5  # Print every 0.5 seconds

    # Performance profiling
    profile_window = 30
    fsm_times = []
    build_action_times = []
    send_action_times = []
    get_obs_times = []
    send_obs_times = []

    while not shutdown_event.is_set():
        loop_start = time.perf_counter()

        # FSM step
        t0 = time.perf_counter()
        fsm.step()
        fsm_times.append(time.perf_counter() - t0)

        # Build action
        t0 = time.perf_counter()
        action = controller.build_action()
        build_action_times.append(time.perf_counter() - t0)

        # Send action
        if action:
            t0 = time.perf_counter()
            robot.send_action(action)
            send_action_times.append(time.perf_counter() - t0)

        # Get observation
        t0 = time.perf_counter()
        observation = robot.get_observation()
        get_obs_times.append(time.perf_counter() - t0)

        # Send observation conditionally to reduce network overhead
        must_go = action is None
        if controller.should_send_observation() or must_go:
            t0 = time.perf_counter()
            controller.send_observation(observation, timestep=timestep, must_go=must_go)
            send_obs_times.append(time.perf_counter() - t0)

        timestep += 1

        elapsed = time.perf_counter() - loop_start
        busy_wait(max(0.0, period - elapsed))

        # Track FPS
        total_time = time.perf_counter() - loop_start
        loop_times.append(total_time)
        if len(loop_times) > fps_window_size:
            loop_times.pop(0)

        # Trim profiling lists
        for lst in [fsm_times, build_action_times, send_action_times, get_obs_times, send_obs_times]:
            while len(lst) > profile_window:
                lst.pop(0)

        # Print FPS and profiling info
        current_time = time.perf_counter()
        if current_time - last_print_time >= print_interval:
            avg_loop_time = sum(loop_times) / len(loop_times)
            current_fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

            # Calculate average times (in milliseconds)
            avg_fsm = sum(fsm_times) / len(fsm_times) * 1000 if fsm_times else 0
            avg_build = sum(build_action_times) / len(build_action_times) * 1000 if build_action_times else 0
            avg_send_act = sum(send_action_times) / len(send_action_times) * 1000 if send_action_times else 0
            avg_get_obs = sum(get_obs_times) / len(get_obs_times) * 1000 if get_obs_times else 0
            avg_send_obs = sum(send_obs_times) / len(send_obs_times) * 1000 if send_obs_times else 0

            print(
                f"\rFPS: {current_fps:.1f}/{target_hz:.0f} | "
                f"FSM: {avg_fsm:.1f}ms | Build: {avg_build:.1f}ms | SendAct: {avg_send_act:.1f}ms | "
                f"GetObs: {avg_get_obs:.1f}ms | SendObs: {avg_send_obs:.1f}ms | Step: {timestep}",
                end='', flush=True
            )
            last_print_time = current_time


@draccus.wrap()
def pick_place_control_loop(cfg: PickPlaceLoopConfig) -> None:
    logging.info("Starting pick-place control loop")

    robot = LeKiwiClient(cfg.robot)
    robot.connect()
    controller = PolicyServerPickPlaceController(cfg.controller, robot)
    fsm = LeKiwiPickPlaceFSM(controller=controller, config=cfg.fsm)

    if cfg.auto_start:
        fsm.step(PickPlaceCommand.START)

    shutdown_event = threading.Event()
    keyboard_thread = None

    try:
        controller.start()

        # Start keyboard listener thread if enabled
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
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fsm_{timestamp}.log"

    # Configure logging with both console and file handlers
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file),  # File output
        ]
    )
    logging.info(f"Log file: {log_file}")
    pick_place_control_loop()


if __name__ == "__main__":
    main()
