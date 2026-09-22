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

"""Static contract tests for the native RLinf YAM example configs."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "env"
    / "realworld_dual_yam_joint.yaml"
)
_COLLECT_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "embodiment"
    / "config"
    / "realworld_dual_yam_collect_data.yaml"
)
_INSTALL_SCRIPT = _REPO_ROOT / "requirements" / "install.sh"
_YAM_REQUIREMENTS = _REPO_ROOT / "requirements" / "embodied" / "envs" / "yam.txt"
_YAM_BUILD_CONSTRAINTS = (
    _REPO_ROOT / "requirements" / "embodied" / "envs" / "yam-build-constraints.txt"
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_dual_yam_env_example_exposes_the_canonical_contract():
    config = _load(_ENV_CONFIG)

    assert config["env_type"] == "realworld"
    assert config["total_num_envs"] == 1
    assert config["main_image_key"] == "top_rgb"
    assert config["init_params"]["id"] == "DualYamJointEnv-v1"
    override = config["override_cfg"]
    assert len(override["joint_limit_min"]) == 2
    assert all(len(limits) == 6 for limits in override["joint_limit_min"])
    assert len(override["joint_limit_max"]) == 2
    assert all(len(limits) == 6 for limits in override["joint_limit_max"])
    assert override["reset"]["enabled"] is False
    assert override["reset"]["mode"] == "startup"
    assert override["reset"]["left_qpos"] is None
    assert override["reset"]["right_qpos"] is None
    assert override["leader_intervention"]["enabled"] is False
    assert override["leader_intervention"]["unsynced_action_source"] == "policy"


def test_dual_yam_collection_example_declares_one_complete_station():
    config = _load(_COLLECT_CONFIG)
    hardware = config["cluster"]["node_groups"][0]["hardware"]

    assert hardware["type"] == "DualYam"
    assert len(hardware["configs"]) == 1
    station = hardware["configs"][0]
    devices = [
        station["left_follower"],
        station["right_follower"],
        station["left_leader"],
        station["right_leader"],
    ]
    assert len({device["channel"] for device in devices}) == 4
    assert station["left_follower"]["gripper_limits"] is None
    assert station["right_follower"]["gripper_limits"] is None
    assert len(station["left_leader"]["gripper_limits"]) == 2
    assert len(station["right_leader"]["gripper_limits"]) == 2
    assert station["left_leader"]["gripper_invert"] is False
    assert station["right_leader"]["gripper_invert"] is False
    assert [camera["name"] for camera in station["cameras"]] == [
        "top_rgb",
        "left_rgb",
        "right_rgb",
    ]
    assert [camera["serial"] for camera in station["cameras"]] == [
        "260322277483",
        "260322272602",
        "260422273719",
    ]

    eval_config = config["env"]["eval"]
    intervention = eval_config["override_cfg"]["leader_intervention"]
    collection = eval_config["data_collection"]
    assert eval_config["max_episode_steps"] == 10000
    assert eval_config["override_cfg"]["manual_episode_control_only"] is True
    assert eval_config["override_cfg"]["enforce_runtime_joint_limits"] is False
    assert eval_config["override_cfg"]["reset"]["mode"] == "manual"
    assert intervention["foot_switch_reset_key"] == 48
    assert intervention["enabled"] is True
    assert intervention["preserve_sync_between_episodes"] is True
    assert intervention["unsynced_action_source"] == "hold"
    assert collection["export_format"] == "lerobot"
    assert collection["robot_type"] == "dual_yam"
    assert collection["fps"] == 30
    # Streaming collection finalizes each kept episode in its own shard.
    assert collection["streaming"] is True
    assert collection["only_success"] is False
    # The RLinf demo replay buffer is skipped so raw images are not buffered
    # in memory a second time.
    assert config["runner"]["save_demos"] is False


def test_dual_yam_examples_have_no_external_application_repo_dependency():
    example_text = _ENV_CONFIG.read_text(encoding="utf-8") + _COLLECT_CONFIG.read_text(
        encoding="utf-8"
    )

    assert "yam-abc-reproduce" not in example_text
    assert "yam_abc_reproduce" not in example_text


def test_yam_install_target_bundles_the_pinned_i2rt_sdk():
    install_text = _INSTALL_SCRIPT.read_text(encoding="utf-8")
    requirements_text = _YAM_REQUIREMENTS.read_text(encoding="utf-8")
    build_constraints_text = _YAM_BUILD_CONSTRAINTS.read_text(encoding="utf-8")

    assert "yam)" in install_text
    assert "install_yam_env" in install_text
    assert "embodied/envs/yam.txt" in install_text
    assert "embodied/envs/yam-build-constraints.txt" in install_text
    assert "yam-abc-reproduce" not in install_text.lower()
    assert "i2rt @ git+https://github.com/i2rt-robotics/i2rt.git@" in requirements_text
    assert "47fee5e7dec4e30ca054f798bda1c8894b465ed2" in requirements_text
    assert "yam-abc-reproduce" not in requirements_text.lower()
    assert "scikit-build-core<0.10" in build_constraints_text


def test_pico_example_composes_into_a_valid_lazy_yam_station(monkeypatch):
    import gymnasium as gym
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    import rlinf.envs.real.yam  # noqa: F401 - registers the Gymnasium ID
    from rlinf.envs.real.yam import DualYamJointEnv, YamPicoConfig, YamPicoEpisode
    from rlinf.robotics import RobotInfo
    from rlinf.robotics.robots.dual_yam import DualYamConfig

    monkeypatch.setenv("EMBODIED_PATH", str(_REPO_ROOT / "examples/embodiment"))
    for role in ("TOP", "LEFT", "RIGHT"):
        monkeypatch.setenv(f"YAM_{role}_CAMERA_SERIAL", f"camera-{role}")
    with initialize_config_dir(
        config_dir=str(_COLLECT_CONFIG.parent), version_base=None
    ):
        cfg = compose(config_name="realworld_dual_yam_collect_data_pico")
    station = OmegaConf.to_container(
        cfg.cluster.node_groups[0].hardware.configs[0], resolve=True
    )
    hardware = RobotInfo(
        type="DualYam", model="DualYam", config=DualYamConfig(**station)
    )
    eval_cfg = OmegaConf.to_container(cfg.env.eval, resolve=True)
    override = eval_cfg["override_cfg"]
    override["is_dummy"] = True
    assert override["enforce_runtime_joint_limits"]
    assert not override["leader_intervention"]["enabled"]
    # The VR recipe selects the paired device through the shared key, and the
    # device-specific options the wrapper reads live under 'pico'.
    assert eval_cfg["teleop"] == "yam_pico"
    assert "use_pico" not in eval_cfg
    assert eval_cfg["pico"]["hand"] == "dual"
    assert eval_cfg["data_collection"]["export_format"] == "lerobot"
    # Same streaming/park contract as the leader-arm collection recipe.
    assert eval_cfg["data_collection"]["streaming"] is True
    assert eval_cfg["data_collection"]["only_success"] is False
    assert eval_cfg["data_collection"]["finalize_interval"] == 10
    assert cfg.runner.save_demos is False
    park = override["park_on_close"]
    assert park["enabled"] is True
    assert len(park["left_qpos"]) == 7 and len(park["right_qpos"]) == 7
    env = gym.make(
        "DualYamJointEnv-v1",
        override_cfg=override,
        worker_info=None,
        robot_info=hardware,
        env_idx=0,
        env_cfg=eval_cfg,
    )
    # A dummy station composes without opening ZMQ, CAN, or cameras.
    assert env.action_space.shape == (14,)
    env.close()

    # The recipe reaches the PICO controller: on a station that does own
    # hardware, the episode control it installs carries the same 'pico' options
    # the device does, so both halves read one block of the YAML.
    station_env = DualYamJointEnv(
        {**override, "is_dummy": False},
        worker_info=None,
        robot_info=hardware,
        env_idx=0,
    )
    (episode_control,) = station_env.episode_wrappers(eval_cfg)
    assert episode_control.func is YamPicoEpisode
    pico_config = episode_control.keywords["config"]
    assert isinstance(pico_config, YamPicoConfig)
    assert pico_config.zmq_addr == eval_cfg["pico"]["zmq_addr"]
    assert pico_config.record_button == eval_cfg["pico"]["record_button"]
    assert pico_config.wait_for_record_button is True
