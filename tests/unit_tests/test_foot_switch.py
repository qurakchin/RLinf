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

"""Unit tests for the evdev foot-switch review input source."""

from __future__ import annotations

import errno
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

EV_KEY = 1
DISCARD_KEY = 30
KEEP_KEY = 46
OTHER_KEY = 48
RESET_KEY = 99


@dataclass
class Event:
    type: int
    code: int
    value: int


class FakeDevice:
    instances = []
    grab_error = None
    active_error = None
    read_error = None
    initial_active = set()

    def __init__(self, path):
        self.path = path
        self.events = []
        self.active = set(FakeDevice.initial_active)
        self.grabbed = False
        self.closed = False
        self.ungrab_calls = 0
        self.close_calls = 0
        FakeDevice.instances.append(self)

    def grab(self):
        if FakeDevice.grab_error is not None:
            raise FakeDevice.grab_error
        self.grabbed = True

    def ungrab(self):
        self.ungrab_calls += 1
        self.grabbed = False

    def close(self):
        self.close_calls += 1
        self.closed = True

    def active_keys(self):
        if FakeDevice.active_error is not None:
            raise FakeDevice.active_error
        return list(self.active)

    def read(self):
        if FakeDevice.read_error is not None:
            raise FakeDevice.read_error
        if not self.events:
            raise BlockingIOError()
        events = self.events
        self.events = []
        return events


@pytest.fixture(autouse=True)
def reset_fake_device():
    FakeDevice.instances = []
    FakeDevice.grab_error = None
    FakeDevice.active_error = None
    FakeDevice.read_error = None
    FakeDevice.initial_active = set()


@pytest.fixture
def foot_switch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "evdev",
        SimpleNamespace(
            InputDevice=FakeDevice,
            ecodes=SimpleNamespace(EV_KEY=EV_KEY),
        ),
    )
    source = (
        Path(__file__).resolve().parents[2] / "rlinf/envs/real/utils/foot_switch.py"
    )
    spec = importlib.util.spec_from_file_location("_foot_switch_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FootSwitch


def make_switch(foot_switch):
    return foot_switch("/dev/input/by-id/fake", DISCARD_KEY, KEEP_KEY)


def make_reset_switch(foot_switch):
    return foot_switch("/dev/input/by-id/fake", DISCARD_KEY, KEEP_KEY, RESET_KEY)


def test_grab_error_closes_opened_device(foot_switch):
    FakeDevice.grab_error = OSError(errno.EBUSY, "busy")

    with pytest.raises(RuntimeError, match="failed to grab.*busy"):
        make_switch(foot_switch)

    assert FakeDevice.instances[-1].close_calls == 1
    assert FakeDevice.instances[-1].ungrab_calls == 0


def test_active_key_error_ungrabs_and_closes_device(foot_switch):
    FakeDevice.active_error = OSError(errno.ENODEV, "gone")

    with pytest.raises(RuntimeError, match="disconnected.*active keys"):
        make_switch(foot_switch)

    assert FakeDevice.instances[-1].ungrab_calls == 1
    assert FakeDevice.instances[-1].close_calls == 1


def test_reset_key_must_be_distinct_from_choice_keys(foot_switch):
    with pytest.raises(ValueError, match="reset_key must be different"):
        foot_switch("/dev/input/by-id/fake", DISCARD_KEY, KEEP_KEY, DISCARD_KEY)


def test_reset_press_reports_once_outside_review_until_release(foot_switch):
    switch = make_reset_switch(foot_switch)
    device = FakeDevice.instances[-1]

    device.events = [Event(EV_KEY, RESET_KEY, 1)]
    assert switch.poll() == "reset"

    device.events = [Event(EV_KEY, RESET_KEY, 2)]
    assert switch.poll() is None
    assert switch.poll() is None

    device.events = [
        Event(EV_KEY, RESET_KEY, 0),
        Event(EV_KEY, RESET_KEY, 1),
    ]
    assert switch.poll() == "reset"


def test_reset_held_at_construction_is_blocked_until_release(foot_switch):
    FakeDevice.initial_active = {RESET_KEY}
    switch = make_reset_switch(foot_switch)
    device = FakeDevice.instances[-1]

    device.events = [Event(EV_KEY, RESET_KEY, 1)]
    assert switch.poll() is None

    device.events = [
        Event(EV_KEY, RESET_KEY, 0),
        Event(EV_KEY, RESET_KEY, 1),
    ]
    assert switch.poll() == "reset"


def test_repeat_and_stale_held_press_are_ignored_until_release(foot_switch):
    switch = make_switch(foot_switch)
    device = FakeDevice.instances[-1]
    device.active = {KEEP_KEY}

    switch.begin_review()
    device.events = [
        Event(EV_KEY, KEEP_KEY, 2),
        Event(EV_KEY, KEEP_KEY, 1),
    ]
    assert switch.poll() is None

    device.events = [
        Event(EV_KEY, KEEP_KEY, 0),
        Event(EV_KEY, KEEP_KEY, 1),
    ]
    assert switch.poll() == "keep"


def test_unrelated_middle_key_and_old_buffered_press_do_not_choose(foot_switch):
    switch = make_switch(foot_switch)
    device = FakeDevice.instances[-1]
    device.events = [
        Event(EV_KEY, KEEP_KEY, 1),
        Event(EV_KEY, OTHER_KEY, 1),
    ]

    switch.begin_review()
    assert switch.poll() is None

    device.events = [Event(EV_KEY, DISCARD_KEY, 1)]
    assert switch.poll() == "discard"
    device.events = [Event(EV_KEY, KEEP_KEY, 1)]
    assert switch.poll() is None


def test_both_choice_keys_are_rejected_until_both_are_released(foot_switch):
    switch = make_switch(foot_switch)
    device = FakeDevice.instances[-1]

    switch.begin_review()
    device.events = [
        Event(EV_KEY, DISCARD_KEY, 1),
        Event(EV_KEY, KEEP_KEY, 1),
    ]
    assert switch.poll() is None

    device.events = [Event(EV_KEY, DISCARD_KEY, 0)]
    assert switch.poll() is None
    device.events = [
        Event(EV_KEY, KEEP_KEY, 0),
        Event(EV_KEY, DISCARD_KEY, 1),
    ]
    assert switch.poll() == "discard"


def test_reset_key_does_not_trigger_when_pressed_with_choice(foot_switch):
    switch = make_reset_switch(foot_switch)
    device = FakeDevice.instances[-1]

    device.events = [
        Event(EV_KEY, RESET_KEY, 1),
        Event(EV_KEY, KEEP_KEY, 1),
    ]
    assert switch.poll() is None

    device.events = [
        Event(EV_KEY, RESET_KEY, 0),
        Event(EV_KEY, KEEP_KEY, 0),
        Event(EV_KEY, RESET_KEY, 1),
    ]
    assert switch.poll() == "reset"


def test_reset_key_does_not_trigger_when_choice_overlap_ends_in_same_poll(
    foot_switch,
):
    switch = make_reset_switch(foot_switch)
    device = FakeDevice.instances[-1]

    device.events = [
        Event(EV_KEY, RESET_KEY, 1),
        Event(EV_KEY, KEEP_KEY, 1),
        Event(EV_KEY, RESET_KEY, 0),
        Event(EV_KEY, KEEP_KEY, 0),
    ]
    assert switch.poll() is None

    device.events = [Event(EV_KEY, RESET_KEY, 1)]
    assert switch.poll() == "reset"


def test_reset_during_review_does_not_consume_keep_or_discard(foot_switch):
    switch = make_reset_switch(foot_switch)
    device = FakeDevice.instances[-1]

    switch.begin_review()
    device.events = [Event(EV_KEY, RESET_KEY, 1)]
    assert switch.poll() == "reset"

    device.events = [
        Event(EV_KEY, RESET_KEY, 0),
        Event(EV_KEY, KEEP_KEY, 1),
    ]
    assert switch.poll() == "keep"


def test_poll_raises_clear_error_on_disconnect(foot_switch):
    switch = make_switch(foot_switch)
    switch.begin_review()
    FakeDevice.read_error = OSError(errno.ENODEV, "gone")

    with pytest.raises(RuntimeError, match="disconnected.*reading"):
        switch.poll()
