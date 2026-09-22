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

"""Launcher contract tests for real-world YAM collection scripts."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EMBODIMENT_DIR = _REPO_ROOT / "examples" / "embodiment"
_COLLECT_DATA_SCRIPT = _EMBODIMENT_DIR / "collect_data.sh"
_CONFIG_NAME = "realworld_dual_yam_collect_data"
_EPISODE_OVERRIDE = "runner.num_data_episodes=90"
_LEFT_PEPSI_TASK = (
    "Tidy up the table. Left and right refer to the top camera view. "
    "Sort all three bottles by brand. "
    "Place all Pepsi bottles in the bag on the left and all Coca-Cola bottles "
    "in the bag on the right. Move the bowls to uncover the spoons, place the "
    "white spoon in the white bowl and the pink spoon in the pink bowl, and "
    "return both bowls to their original marked positions: white bowl on the "
    "left and pink bowl on the right."
)
_RIGHT_PEPSI_TASK = _LEFT_PEPSI_TASK.replace(
    "Place all Pepsi bottles in the bag on the left and all Coca-Cola bottles "
    "in the bag on the right.",
    "Place all Pepsi bottles in the bag on the right and all Coca-Cola bottles "
    "in the bag on the left.",
)
_TASK_OVERRIDE = f'env.eval.override_cfg.task_description="{_LEFT_PEPSI_TASK}"'
_LAUNCHER_CASES = (
    (
        "collect_tabletop_cleanup_1.sh",
        1,
        _LEFT_PEPSI_TASK,
        "[入袋规则] 左袋：百事可乐；右袋：可口可乐",
        "[初始勺子] 左碗后：粉勺；右碗后：白勺",
    ),
    (
        "collect_tabletop_cleanup_2.sh",
        2,
        _RIGHT_PEPSI_TASK,
        "[入袋规则] 左袋：可口可乐；右袋：百事可乐",
        "[初始勺子] 左碗后：粉勺；右碗后：白勺",
    ),
)


def _copy_embodiment_script(
    tmp_path: Path, source: Path = _COLLECT_DATA_SCRIPT
) -> Path:
    """Copy a launcher into a tiny repo layout so it resolves paths normally."""
    embodiment_dir = tmp_path / "repo" / "examples" / "embodiment"
    script_path = embodiment_dir / source.name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    if source != _COLLECT_DATA_SCRIPT:
        common_script = embodiment_dir / _COLLECT_DATA_SCRIPT.name
        shutil.copy2(_COLLECT_DATA_SCRIPT, common_script)
        common_script.chmod(0o755)
    shutil.copy2(source, script_path)
    script_path.chmod(0o755)
    return script_path


def _install_fake_python(tmp_path: Path, exit_code: int = 0) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_path = tmp_path / "python-argv.bin"
    python_path = bin_dir / "python"
    python_path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                ': > "$FAKE_PYTHON_ARGV"',
                'for arg in "$@"; do',
                '    printf "%s\\0" "$arg" >> "$FAKE_PYTHON_ARGV"',
                "done",
                f"exit {exit_code}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    return bin_dir, argv_path


def _run_launcher(
    script_path: Path,
    tmp_path: Path,
    args: list[str],
    *,
    exit_code: int = 0,
    log_dir: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    bin_dir, argv_path = _install_fake_python(tmp_path, exit_code=exit_code)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_PYTHON_ARGV": str(argv_path),
    }
    if log_dir is not None:
        env["RLINF_LOG_DIR"] = str(log_dir)

    result = subprocess.run(
        ["bash", str(script_path), *args],
        cwd=tmp_path / "repo",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    raw_argv = argv_path.read_bytes() if argv_path.exists() else b""
    argv = [part.decode() for part in raw_argv.split(b"\0") if part]
    return result, argv


def _logged_path(argv: list[str]) -> Path:
    prefix = "runner.logger.log_path="
    (override,) = [arg for arg in argv if arg.startswith(prefix)]
    return Path(override.removeprefix(prefix))


def test_collect_data_uses_config_argument_and_forwards_hydra_overrides(tmp_path):
    script_path = _copy_embodiment_script(tmp_path)
    custom_log_dir = tmp_path / "custom collection logs"

    result, argv = _run_launcher(
        script_path,
        tmp_path,
        [
            _CONFIG_NAME,
            _EPISODE_OVERRIDE,
            _TASK_OVERRIDE,
            "env.eval.data_collection.only_success=false",
        ],
        log_dir=custom_log_dir,
    )

    assert result.returncode == 0, result.stderr
    log_dir = _logged_path(argv)
    assert argv == [
        str(tmp_path / "repo" / "examples" / "embodiment" / "collect_real_data.py"),
        "--config-path",
        str(tmp_path / "repo" / "examples" / "embodiment" / "config") + "/",
        "--config-name",
        _CONFIG_NAME,
        f"runner.logger.log_path={log_dir}",
        _EPISODE_OVERRIDE,
        _TASK_OVERRIDE,
        "env.eval.data_collection.only_success=false",
    ]
    log_file = log_dir / "run_embodiment.log"
    assert log_file.exists()
    assert shlex.split(log_file.read_text(encoding="utf-8")) == ["python", *argv]


def test_collect_data_propagates_python_failure_through_tee(tmp_path):
    script_path = _copy_embodiment_script(tmp_path)

    result, argv = _run_launcher(
        script_path,
        tmp_path,
        [_CONFIG_NAME, _EPISODE_OVERRIDE],
        exit_code=37,
        log_dir=tmp_path / "logs",
    )

    assert result.returncode == 37
    assert argv[-1] == _EPISODE_OVERRIDE


@pytest.mark.parametrize(
    ("launcher_name", "variant", "task_text", "bag_hint", "spoon_hint"),
    _LAUNCHER_CASES,
)
def test_tabletop_cleanup_launchers_forward_collection_contract(
    tmp_path: Path,
    launcher_name: str,
    variant: int,
    task_text: str,
    bag_hint: str,
    spoon_hint: str,
):
    source = _EMBODIMENT_DIR / launcher_name
    assert source.exists(), f"missing tabletop cleanup launcher: {launcher_name}"
    script_path = _copy_embodiment_script(tmp_path, source=source)

    forwarded_override = "runner.num_data_episodes=7"
    result, argv = _run_launcher(script_path, tmp_path, [forwarded_override])

    assert result.returncode == 0, result.stderr
    log_dir = _logged_path(argv)
    assert "--config-name" in argv
    assert argv[argv.index("--config-name") + 1] == _CONFIG_NAME
    assert _EPISODE_OVERRIDE in argv
    assert argv[-1] == forwarded_override
    task_overrides = [
        arg for arg in argv if arg.startswith("env.eval.override_cfg.task_description=")
    ]
    assert len(task_overrides) == 1
    task_description = task_overrides[0].removeprefix(
        "env.eval.override_cfg.task_description="
    )
    assert task_description == f'"{task_text}"'
    assert "左碗后" not in task_description
    assert "右碗后" not in task_description
    assert f"runner.logger.log_path={log_dir}" in argv
    assert log_dir.parent.name == f"variant_{variant}"
    assert log_dir.parent.parent == tmp_path / "repo" / "logs" / "tabletop_cleanup"
    log_file = log_dir / "run_embodiment.log"
    assert log_file.exists()
    assert log_file.parent == log_dir
    assert bag_hint in result.stdout
    assert spoon_hint in result.stdout


def test_tabletop_cleanup_launcher_respects_log_dir_override(tmp_path: Path):
    source = _EMBODIMENT_DIR / "collect_tabletop_cleanup_1.sh"
    assert source.exists(), (
        "missing tabletop cleanup launcher: collect_tabletop_cleanup_1.sh"
    )
    script_path = _copy_embodiment_script(tmp_path, source=source)
    custom_log_dir = tmp_path / "custom tabletop logs"

    result, argv = _run_launcher(script_path, tmp_path, [], log_dir=custom_log_dir)

    assert result.returncode == 0, result.stderr
    assert _logged_path(argv) == custom_log_dir
    assert f"[输出目录] {custom_log_dir}" in result.stdout
