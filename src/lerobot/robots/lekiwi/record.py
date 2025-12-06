# !/usr/bin/env python

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

import shutil
from pathlib import Path

from datasets.exceptions import DatasetGenerationError
import pyarrow.lib as pa_lib

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR, HF_LEROBOT_HOME
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 3000
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick the green object and put it on the white"
HF_REPO_ID = "khmin101/poc2_TC_place"

# Create the robot and teleoperator configurations
robot_config = LeKiwiClientConfig(remote_ip="192.168.20.199", id="lekiwi")
leader_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem5AAF2199371", id="so101_follower")
keyboard_config = KeyboardTeleopConfig()

# Initialize the robot and teleoperator
robot = LeKiwiClient(robot_config)
leader_arm = SO100Leader(leader_arm_config)
keyboard = KeyboardTeleop(keyboard_config)

# TODO(Steven): Update this example to use pipelines
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
dataset_features = {**action_features, **obs_features}


def backup_dataset_cache(dataset_root: Path, label: str) -> None:
    backup_root = dataset_root.parent / f"{dataset_root.name}_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / label
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(dataset_root, backup_dir)
    print(f"Cached dataset backup saved to {backup_dir}")

# Create or resume the dataset
dataset_root = HF_LEROBOT_HOME / HF_REPO_ID
resume_recording = dataset_root.exists()

if resume_recording:
    print(f"Resuming dataset '{HF_REPO_ID}' from {dataset_root}")
    dataset = LeRobotDataset(repo_id=HF_REPO_ID)
else:
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

# Connect the robot and teleoperator
# To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
robot.connect()
leader_arm.connect()
keyboard.connect()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="lekiwi_record")

if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting record loop...")
recorded_episodes = dataset.meta.total_episodes if resume_recording else 0
target_episode = recorded_episodes + NUM_EPISODES
if resume_recording:
    print(f"Continuing from episode {recorded_episodes}, target {target_episode}")

while recorded_episodes < target_episode and not events["stop_recording"]:
    print(f"Recording episode {recorded_episodes}")

    # Main record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=[leader_arm, keyboard],
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (
        (recorded_episodes < target_episode - 1) or events["rerecord_episode"]
    ):
        print("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=[leader_arm, keyboard],
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    if events["rerecord_episode"]:
        print("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # Save episode
    dataset.save_episode()
    recorded_episodes += 1

# Clean up
print("Stop recording")
robot.disconnect()
leader_arm.disconnect()
keyboard.disconnect()
listener.stop()

dataset.finalize()
backup_dataset_cache(dataset_root, f"post_finalize_{recorded_episodes:04d}")
dataset.push_to_hub()
