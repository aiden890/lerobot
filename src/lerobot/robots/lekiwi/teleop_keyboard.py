#!/usr/bin/env python3
"""
Simple keyboard teleoperation loop for LeKiwiClient.

Usage:
    python teleop_keyboard.py --remote-ip 192.168.0.50

The default key bindings are identical to `LeKiwiClientConfig.teleop_keys`:
W/A/S/D move the base in the plane, Z/X rotate, R/F change speed tier, and Q exits.
"""

import argparse
import sys
import time
from threading import Event, Lock

import numpy as np
from pynput import keyboard

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard teleop helper for LeKiwi.")
    parser.add_argument("--remote-ip", required=True, help="IP address of the Jetson running lekiwi_host.")
    parser.add_argument("--cmd-port", type=int, default=5555, help="ZeroMQ command port (default: 5555).")
    parser.add_argument(
        "--obs-port", type=int, default=5556, help="ZeroMQ observation port (default: 5556)."
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Command frequency in Hz (default: 20).")
    return parser.parse_args()


class KeyboardTeleop:
    def __init__(self, client: LeKiwiClient, loop_rate_hz: float):
        self.client = client
        self.loop_rate_hz = loop_rate_hz
        self._pressed = set()
        self._lock = Lock()
        self._stop = Event()
        self._quit_key = self.client.teleop_keys.get("quit", "q")

    def _on_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        with self._lock:
            self._pressed.add(char)

        if char == self._quit_key:
            print("Quit key pressed, stopping teleop.")
            self._stop.set()

    def _on_release(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return

        with self._lock:
            self._pressed.discard(char)

    def run(self):
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        dt = 1.0 / self.loop_rate_hz

        print("Keyboard teleop running. Hold Q (or configured quit key) to stop.")

        try:
            while not self._stop.is_set():
                with self._lock:
                    pressed = np.array(list(self._pressed))
                base_action = self.client._from_keyboard_to_base_action(pressed)
                self.client.send_action(base_action)
                time.sleep(dt)
        finally:
            listener.stop()


def main():
    if sys.platform == "darwin":
        print("macOS 사용자는 시스템 설정 > 개인정보 보호 및 보안 > 입력 모니터링에서 터미널 권한을 허용하세요.")
    args = parse_args()

    config = LeKiwiClientConfig(remote_ip=args.remote_ip, port_zmq_cmd=args.cmd_port, port_zmq_observations=args.obs_port)
    client = LeKiwiClient(config)
    client.connect()

    teleop = KeyboardTeleop(client, loop_rate_hz=args.rate)
    try:
        teleop.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping teleop.")
    finally:
        client.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        client.disconnect()


if __name__ == "__main__":
    main()
