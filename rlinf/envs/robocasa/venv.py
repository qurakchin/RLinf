# Copyright 2025 The RLinf Authors.
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

"""Subprocess vectorized environment for Robocasa.

Based on metaworld/venv.py implementation, adapted for Robocasa/Robosuite environments.
"""

import inspect
import re
from multiprocessing import Pipe, connection
from multiprocessing.context import Process
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)


def _render_raw(env, cam, h, w, depth):
    """RoboCasa-style raw render.

    Returns ``rgb`` (uint8 HxWx3) in robosuite-native orientation, and when
    ``depth`` is set, *metric* depth (HxW) post-processed via
    ``CU.get_real_depth_map`` so it back-projects to world space.
    """
    import robosuite.utils.camera_utils as CU

    out = env.sim.render(width=w, height=h, camera_name=cam, depth=depth)
    if depth:
        rgb, d = out
        # Sanitize raw OpenGL normalized depth into [0,1]: replace NaN/inf
        # (degenerate camera pose) then clip numerical overshoot. Otherwise
        # get_real_depth_map asserts and kills the worker process.
        d = np.nan_to_num(d, nan=1.0, posinf=1.0, neginf=0.0)
        d = np.clip(d, 0.0, 1.0)
        if d.ndim == 3:
            depth_out = CU.get_real_depth_map(env.sim, d)[..., 0]
        else:
            depth_out = CU.get_real_depth_map(env.sim, d[..., None])[..., 0]
        return rgb, depth_out
    return out


def _get_camera_meta(env, camera_name, height=None, width=None):
    """Static camera calibration (K, cam→world extrinsics, depth near/far)."""
    import robosuite.utils.camera_utils as CU

    K = CU.get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    Ext = CU.get_camera_extrinsic_matrix(env.sim, camera_name)  # cam->world
    m = env.sim.model
    extent = m.stat.extent
    return {
        "camera_name": camera_name,
        "height": height,
        "width": width,
        "intrinsic": np.asarray(K, dtype=np.float64).tolist(),
        "extrinsic_cam2world": np.asarray(Ext, dtype=np.float64).tolist(),
        "depth_near": float(m.vis.map.znear * extent),
        "depth_far": float(m.vis.map.zfar * extent),
    }


def _get_camera_transform(env, camera_name, height=None, width=None):
    """Pixel-to-world 4x4 (``inv(T_world2cam)``) for the named camera."""
    import robosuite.utils.camera_utils as CU

    T = CU.get_camera_transform_matrix(env.sim, camera_name, height, width)
    return np.linalg.inv(T)  # T_p2w


def _grasp_contact(env):
    """Direction-AGNOSTIC grasp check: True iff BOTH gripper fingerpads are in
    contact with the SAME task object. Returns ``(contacting, object_name)``."""
    try:
        grip = env.robots[0].gripper  # {"right": GripperModel}
        for name, obj in env.objects.items():
            try:
                if env._check_grasp(grip, obj):
                    return True, name
            except Exception:
                continue
    except Exception:
        pass
    return False, None


def _reassemble_env_action(env, unmap_result):
    """Reassemble an RLDX ``unmap_result`` dict into a flat robosuite action."""
    from robosuite.controllers.composite.composite_controller import (
        HybridMobileBase,
    )

    env_action = []
    for robot in env.robots:
        cc = robot.composite_controller
        pf = robot.robot_model.naming_prefix
        a = np.zeros(cc.action_limits[0].shape)
        for part_name in cc.part_controllers:
            s, e = cc._action_split_indexes[part_name]
            a[s:e] = unmap_result.pop(f"{pf}{part_name}")
        if isinstance(cc, HybridMobileBase):
            a[-1] = unmap_result.pop(f"{pf}base_mode")
        env_action.append(a)
    return np.concatenate(env_action)


def _get_success_criteria_text(env):
    """Source of ``_check_success`` + resolved helper fixtures/methods."""
    out = []
    try:
        src = inspect.getsource(type(env)._check_success)
        out.append(
            "# SUCCESS CONDITION for this task (env._check_success)\n"
            "# You must make this return True. Object positions are NOT given —\n"
            "# localize every named object/fixture from the camera+world maps.\n\n"
            + src
        )
        try:
            import robocasa.utils.object_utils as OU

            for fn in sorted(set(re.findall(r"OU\.(\w+)\(", src))):
                f = getattr(OU, fn, None)
                if f is not None:
                    try:
                        out.append("## helper OU.%s\n%s" % (fn, inspect.getsource(f)))
                    except Exception:
                        pass
        except Exception:
            pass
        for fix, meth in sorted(set(re.findall(r"self\.(\w+)\.(\w+)\(", src))):
            obj = getattr(env, fix, None)
            if obj is not None and hasattr(type(obj), meth):
                try:
                    out.append(
                        "## %s.%s\n%s" % (fix, meth, inspect.getsource(getattr(type(obj), meth)))
                    )
                except Exception:
                    pass
    except Exception as ex:
        out.append("(_check_success extraction failed: %s)" % ex)
    return "\n\n".join(out)[:9000]


def _get_task_progress(env):
    """Scalar/bool progress dict mined from ``_check_success`` locals."""
    prog = {}
    code = type(env)._check_success.__code__
    try:
        src = inspect.getsource(type(env)._check_success)
        # Capture both ``self.attr`` AND dotted ``self.fixture._attr`` paths used
        # in the success check — a bare-attr regex would only grab the fixture
        # object and miss the real gating flag (e.g. self.coffee_machine._turned_on).
        for path in sorted(set(re.findall(r"self\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", src))):
            obj = env
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if not ok:
                continue
            key = path.replace(".", "_")
            if isinstance(obj, (bool, np.bool_)):
                prog[key] = bool(obj)
            elif isinstance(obj, (int, np.integer)):
                prog[key] = int(obj)
            elif isinstance(obj, (float, np.floating)):
                prog[key] = round(float(obj), 4)
    except Exception:
        pass
    # Trace ONE read-only call of _check_success; grab its return-frame locals.
    captured = {}

    def _tracer(frame, event, arg):
        if event == "call" and frame.f_code is code:
            def _local(f, e, a):
                if e == "return":
                    captured.update(f.f_locals)
                return _local

            return _local
        return None

    import sys as _sys

    old = _sys.gettrace()
    try:
        _sys.settrace(_tracer)
        env._check_success()
    except Exception:
        pass
    finally:
        _sys.settrace(old)
    for k, v in captured.items():
        if k == "self" or k in prog:
            continue
        if isinstance(v, (bool, np.bool_)):
            prog[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            prog[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            prog[k] = round(float(v), 4)
    return prog


def _set_seed(env, seed):
    """Re-seed the subprocess env (random/np/env.seed/env.rng) — used by the
    RLDX_RESET_SEED paired-comparison flow."""
    import random

    sd = int(seed)
    random.seed(sd)
    np.random.seed(sd)
    if hasattr(env, "seed"):
        env.seed = sd
    if hasattr(env, "rng"):
        env.rng = np.random.default_rng(sd)
    return True


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    """Worker function for robocasa subprocess environment.

    Based on metaworld's _worker function, adapted for robosuite environments.
    """

    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    def _check_success(env, env_return):
        success = env._check_success()
        env_return = list(env_return)
        info = env_return[-1]
        info["success"] = success
        env_return[-1] = info
        env_return = tuple(env_return)
        return env_return

    def get_ep_meta(env, env_return):
        ep_meta = env.get_ep_meta()
        env_return = list(env_return)
        info = env_return[-1]
        info["ep_meta"] = ep_meta
        env_return[-1] = info
        env_return = tuple(env_return)
        return env_return

    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                # Robosuite returns (obs, reward, done, info), not 5 values like gymnasium
                env_return = env.step(data)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                # RoboCasa step can't record success in info, _check_success() must be called
                if hasattr(env, "_check_success"):
                    env_return = _check_success(env, env_return)
                # call get_ep_meta() to get the RoboCasa env meta, includes prompt & layout_id, etcs
                if hasattr(env, "get_ep_meta"):
                    env_return = get_ep_meta(env, env_return)
                p.send(env_return)
            elif cmd == "reset":
                # Robosuite reset can return just obs or (obs, info)
                retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                    info = {}
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                # call get_ep_meta() to get the RoboCasa env meta, includes prompt & layout_id, etcs
                if hasattr(env, "get_ep_meta"):
                    info = get_ep_meta(env, (info,))[-1]
                # return obs + info other than mere obs
                p.send((obs, info))
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "reconfigure":
                env.close()
                env = data.data()
                p.send(None)
            elif cmd == "check_success":
                p.send(env._check_success())
            elif cmd == "get_camera_meta":
                p.send(_get_camera_meta(env, **data))
            elif cmd == "render_camera":
                # Render an arbitrary camera at the requested resolution.
                # Falls back to the env's render() when no sim attribute.
                if hasattr(env, "sim"):
                    p.send(_render_raw(
                        env,
                        cam=data["camera_name"],
                        h=data["height"],
                        w=data["width"],
                        depth=data["depth"],
                    ))
                else:
                    p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "render_raw":
                p.send(_render_raw(env, **data))
            elif cmd == "get_camera_transform":
                p.send(_get_camera_transform(env, **data))
            elif cmd == "get_ep_meta":
                p.send(env.get_ep_meta() if hasattr(env, "get_ep_meta") else {})
            elif cmd == "grasp_contact":
                p.send(_grasp_contact(env))
            elif cmd == "reassemble_env_action":
                p.send(_reassemble_env_action(env, data["unmap_result"]))
            elif cmd == "get_success_criteria_text":
                p.send(_get_success_criteria_text(env))
            elif cmd == "get_task_progress":
                p.send(_get_task_progress(env))
            elif cmd == "set_seed":
                p.send(_set_seed(env, data))
            else:
                p.close()
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        p.close()


class RobocasaSubprocEnvWorker(SubprocEnvWorker):
    """Subprocess environment worker for Robocasa.

    Based on metaworld's ReconfigureSubprocEnvWorker, but without the reconfigure
    functionality since robocasa doesn't need it.
    """

    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        # Use our custom _worker function
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def reconfigure_env_fn(self, env_fn: Callable[[], gym.Env]) -> None:
        self.parent_remote.send(["reconfigure", CloudpickleWrapper(env_fn)])
        return self.parent_remote.recv()


class RobocasaSubprocEnv(SubprocVectorEnv):
    """Subprocess vectorized environment for Robocasa/Robosuite.

    Based on metaworld's ReconfigureSubprocEnv, adapted for robocasa environments.
    Uses subprocess isolation to avoid OpenGL context sharing issues in MuJoCo.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> RobocasaSubprocEnvWorker:
            # Use our custom worker with shared memory disabled
            # Robosuite dict observations work better without shared memory
            return RobocasaSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])
