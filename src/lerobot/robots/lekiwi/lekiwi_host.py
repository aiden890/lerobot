#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from .config_lekiwi import LeKiwiConfig, LeKiwiHostConfig
from .lekiwi import LeKiwi


@dataclass
class LeKiwiServerConfig:
    """Configuration for the LeKiwi host script."""

    robot: LeKiwiConfig = field(default_factory=LeKiwiConfig)
    host: LeKiwiHostConfig = field(default_factory=LeKiwiHostConfig)


class LeKiwiHost:
    def __init__(self, config: LeKiwiHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: LeKiwiServerConfig):
    logging.info("Configuring LeKiwi")
    robot = LeKiwi(cfg.robot)

    logging.info("Connecting LeKiwi")
    robot.connect()

    logging.info("Starting HostAgent")
    host = LeKiwiHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")

    # FPS tracking
    fps_window_size = 30
    loop_times = []
    last_print_time = time.perf_counter()
    print_interval = 0.5  # Print every 0.5 seconds

    # Performance profiling
    profile_window = 30
    recv_cmd_times = []
    send_action_times = []
    get_obs_times = []
    encode_times = []
    send_obs_times = []
    timestep = 0

    try:
        # Business logic
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.perf_counter()

            # Receive command
            t0 = time.perf_counter()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                recv_cmd_times.append(time.perf_counter() - t0)

                # Send action
                t0 = time.perf_counter()
                _action_sent = robot.send_action(data)
                send_action_times.append(time.perf_counter() - t0)

                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                recv_cmd_times.append(time.perf_counter() - t0)
                if not watchdog_active:
                    logging.warning("No command available")
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            now = time.time()
            if (now - last_cmd_time > host.watchdog_timeout_ms / 1000) and not watchdog_active:
                logging.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} milliseconds. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            # Get observation
            t0 = time.perf_counter()
            last_observation = robot.get_observation()
            get_obs_times.append(time.perf_counter() - t0)

            # Encode ndarrays to base64 strings
            t0 = time.perf_counter()
            for cam_key, _ in robot.cameras.items():
                ret, buffer = cv2.imencode(
                    ".jpg", last_observation[cam_key], [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                )
                if ret:
                    last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8")
                else:
                    last_observation[cam_key] = ""
            encode_times.append(time.perf_counter() - t0)

            # Send the observation to the remote agent
            t0 = time.perf_counter()
            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")
            send_obs_times.append(time.perf_counter() - t0)

            timestep += 1

            # Ensure a short sleep to avoid overloading the CPU.
            elapsed = time.perf_counter() - loop_start_time

            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))

            # Track FPS
            total_time = time.perf_counter() - loop_start_time
            loop_times.append(total_time)
            if len(loop_times) > fps_window_size:
                loop_times.pop(0)

            # Trim profiling lists
            for lst in [recv_cmd_times, send_action_times, get_obs_times, encode_times, send_obs_times]:
                while len(lst) > profile_window:
                    lst.pop(0)

            # Print FPS and profiling info
            current_time = time.perf_counter()
            if current_time - last_print_time >= print_interval:
                avg_loop_time = sum(loop_times) / len(loop_times)
                current_fps = 1.0 / avg_loop_time if avg_loop_time > 0 else 0

                # Calculate average times (in milliseconds)
                avg_recv = sum(recv_cmd_times) / len(recv_cmd_times) * 1000 if recv_cmd_times else 0
                avg_send_act = sum(send_action_times) / len(send_action_times) * 1000 if send_action_times else 0
                avg_get_obs = sum(get_obs_times) / len(get_obs_times) * 1000 if get_obs_times else 0
                avg_encode = sum(encode_times) / len(encode_times) * 1000 if encode_times else 0
                avg_send_obs = sum(send_obs_times) / len(send_obs_times) * 1000 if send_obs_times else 0

                print(
                    f"\rFPS: {current_fps:.1f}/{host.max_loop_freq_hz:.0f} | "
                    f"RecvCmd: {avg_recv:.1f}ms | SendAct: {avg_send_act:.1f}ms | "
                    f"GetObs: {avg_get_obs:.1f}ms | Encode: {avg_encode:.1f}ms | SendObs: {avg_send_obs:.1f}ms | "
                    f"Step: {timestep}",
                    end='', flush=True
                )
                last_print_time = current_time

            duration = time.perf_counter() - start
        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down Lekiwi Host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
