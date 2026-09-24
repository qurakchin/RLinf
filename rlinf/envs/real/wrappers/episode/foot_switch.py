# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
from typing import Any


class FootSwitch:
    """Non-threaded evdev foot switch for review and reset input."""

    def __init__(
        self,
        device_path: str,
        discard_key: int,
        keep_key: int,
        reset_key: int | None = None,
    ) -> None:
        if discard_key == keep_key:
            raise ValueError("discard_key and keep_key must be different.")
        if reset_key is not None and int(reset_key) in {
            int(discard_key),
            int(keep_key),
        }:
            raise ValueError(
                "reset_key must be different from discard_key and keep_key."
            )
        try:
            from evdev import InputDevice, ecodes
        except ImportError as exc:
            raise RuntimeError(
                "FootSwitch requires the 'evdev' package. Install the real-world "
                "extras with evdev support."
            ) from exc

        self.device_path = device_path
        self.discard_key = int(discard_key)
        self.keep_key = int(keep_key)
        self.reset_key = None if reset_key is None else int(reset_key)
        self._choice_keys = {self.discard_key, self.keep_key}
        self._reset_keys = set() if self.reset_key is None else {self.reset_key}
        self._ev_key = ecodes.EV_KEY
        self.device: Any | None = None
        self._grabbed = False
        self._closed = False
        self._pressed_keys: set[int] = set()
        self._blocked_keys: set[int] = set()
        self._reject_until_release = False
        self._choice_consumed = True
        self._reset_blocked = False

        try:
            self.device = InputDevice(device_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"FootSwitch device path '{device_path}' does not exist."
            ) from exc
        except PermissionError as exc:
            raise RuntimeError(
                f"FootSwitch cannot read device '{device_path}'. Grant the runtime "
                "user read access to the input device."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"FootSwitch failed to open device '{device_path}': {exc}"
            ) from exc

        try:
            self.device.grab()
        except OSError as exc:
            self._cleanup_device()
            raise RuntimeError(
                f"FootSwitch failed to grab device '{device_path}': {exc}"
            ) from exc
        self._grabbed = True
        try:
            self._pressed_keys = self._active_keys()
            self._reset_blocked = bool(self._pressed_keys & self._reset_keys)
        except Exception:
            self._cleanup_device()
            raise

    def begin_review(self) -> None:
        """Clear stale events and require already-held choice keys to cycle."""

        while True:
            events = self._read_events()
            if not events:
                break
            for event in events:
                self._handle_event(event)
        self._pressed_keys = self._active_keys()
        held = self._pressed_keys & self._choice_keys
        self._blocked_keys = set(held)
        self._reject_until_release = len(held) == 2
        self._reset_blocked = bool(self._pressed_keys & self._reset_keys)
        self._choice_consumed = False

    def poll(self) -> str | None:
        """Return 'keep', 'discard', 'reset', or None without blocking."""

        choice = None
        reset_requested = False
        reset_overlapped_choice = False
        choice_press_seen = False
        conflict = False
        for event in self._read_events():
            is_key_event = event.type == self._ev_key
            key = int(event.code) if is_key_event else None
            value = int(event.value) if is_key_event else None
            choice_was_pressed = bool(self._pressed_keys & self._choice_keys)
            choice_press_seen_before = choice_press_seen

            event_choice, event_conflict = self._handle_event(event)
            conflict = conflict or event_conflict

            is_choice_press = key in self._choice_keys and value == 1
            if is_choice_press:
                choice_press_seen = True
                if reset_requested:
                    reset_overlapped_choice = True
            if event_choice == "reset":
                reset_requested = True
                if choice_was_pressed or choice_press_seen_before:
                    reset_overlapped_choice = True
            elif choice is None:
                choice = event_choice

        if reset_requested:
            if conflict or reset_overlapped_choice:
                return None
            return "reset"
        if self._choice_consumed or conflict or self._reject_until_release:
            return None
        if choice is None:
            return None
        self._choice_consumed = True
        return choice

    def close(self) -> None:
        """Ungrab and close the device. Repeated calls are no-ops."""

        if self._closed:
            return
        errors = self._cleanup_device()
        if errors:
            raise RuntimeError(
                f"FootSwitch failed to close device '{self.device_path}': "
                + "; ".join(errors)
            )

    def _read_events(self) -> list[Any]:
        self._check_open()
        try:
            return list(self.device.read())
        except BlockingIOError:
            return []
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return []
            if exc.errno == errno.ENODEV:
                raise RuntimeError(
                    f"FootSwitch device '{self.device_path}' disconnected while "
                    "reading."
                ) from exc
            raise RuntimeError(
                f"FootSwitch failed to read device '{self.device_path}': {exc}"
            ) from exc

    def _active_keys(self) -> set[int]:
        self._check_open()
        try:
            return {int(key) for key in self.device.active_keys()}
        except OSError as exc:
            if exc.errno == errno.ENODEV:
                raise RuntimeError(
                    f"FootSwitch device '{self.device_path}' disconnected while "
                    "checking active keys."
                ) from exc
            raise RuntimeError(
                f"FootSwitch failed to check active keys for '{self.device_path}': "
                f"{exc}"
            ) from exc

    def _handle_event(self, event: Any) -> tuple[str | None, bool]:
        if event.type != self._ev_key:
            return None, False

        key = int(event.code)
        if event.value == 0:
            self._pressed_keys.discard(key)
            self._blocked_keys.discard(key)
            if key == self.reset_key:
                self._reset_blocked = False
            if not (self._pressed_keys & self._choice_keys):
                self._reject_until_release = False
            return None, False
        if event.value != 1:
            return None, False

        self._pressed_keys.add(key)
        if key == self.reset_key:
            if self._reset_blocked or self._pressed_keys & self._choice_keys:
                self._reset_blocked = True
                return None, False
            self._reset_blocked = True
            return "reset", False
        if key not in self._choice_keys:
            return None, False
        if len(self._pressed_keys & self._choice_keys) == 2:
            self._reject_until_release = True
            return None, True
        if key in self._blocked_keys or self._reject_until_release:
            return None, False
        return ("keep" if key == self.keep_key else "discard"), False

    def _cleanup_device(self) -> list[str]:
        device = self.device
        self.device = None
        self._closed = True
        errors = []
        if device is None:
            return errors
        if self._grabbed:
            try:
                device.ungrab()
            except OSError as exc:
                errors.append(f"ungrab failed: {exc}")
            self._grabbed = False
        try:
            device.close()
        except OSError as exc:
            errors.append(f"close failed: {exc}")
        return errors

    def _check_open(self) -> None:
        if self._closed or self.device is None:
            raise RuntimeError(
                f"FootSwitch device '{self.device_path}' is already closed."
            )
