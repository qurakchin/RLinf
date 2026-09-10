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


"""Hardware-free checks for the local processed-camera preview."""

from urllib.request import urlopen

import cv2
import numpy as np

from rlinf.envs.real.utils.camera_preview import CameraPreview


def test_preview_latest_rgb_frames_and_stop():
    preview = CameraPreview(0)
    # Test against the actual ephemeral listener, without cameras or CAN.
    port = preview._server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    try:
        with urlopen(url, timeout=2) as response:
            assert b"frame.jpg" in response.read()
        old = {"top_rgb": np.zeros((48, 64, 3), dtype=np.uint8)}
        preview.put_frame(old)
        frames = {
            name: np.full((48, 64, 3), color, dtype=np.uint8)
            for name, color in zip(
                ("top_rgb", "left_rgb", "right_rgb"),
                ((255, 0, 0), (0, 255, 0), (0, 0, 255)),
            )
        }
        for _ in range(100):
            preview.put_frame(frames)
        with urlopen(url + "/frame.jpg", timeout=2) as response:
            decoded = cv2.imdecode(np.frombuffer(response.read(), np.uint8), 1)
        assert decoded.shape[1] == 960
        # OpenCV decodes BGR: confirm RGB source colors and view ordering.
        for i, expected_channel in enumerate((2, 1, 0)):
            pixel = decoded[decoded.shape[0] // 2, 160 + i * 320]
            assert pixel[expected_channel] > 240
            assert np.count_nonzero(pixel > 20) == 1
        assert (frames["top_rgb"] == [255, 0, 0]).all()
    finally:
        preview.stop()
        preview.stop()
    assert not preview._thread.is_alive()


def test_yam_preview_receives_observations_and_releases_port():
    from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv

    env = DualYamJointEnv({"is_dummy": True, "camera_preview_port": 0})
    assert env._camera_player is None
    try:
        obs, _ = env.reset()
        player = env._camera_player
        url = f"http://127.0.0.1:{player._server.server_address[1]}/frame.jpg"
        with urlopen(url, timeout=2) as response:
            assert response.headers["Content-Type"] == "image/jpeg"
        assert list(obs["frames"]) == ["top_rgb", "left_rgb", "right_rgb"]
        assert all(not frame.any() for frame in obs["frames"].values())
    finally:
        env.close()
    assert not player._thread.is_alive()
