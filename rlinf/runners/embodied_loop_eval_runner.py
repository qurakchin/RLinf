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

"""Eval runner for the EmbodiedLoop + StatelessRollout split.

Wires three worker groups (env / embodied_loop / rollout) via four named channels:
    - "EnvAction":       EmbodiedLoop -> Env   (action chunks)
    - "EnvObs":          Env -> EmbodiedLoop   (obs batches)
    - "RolloutRequest":  EmbodiedLoop -> Rollout  (obs payload)
    - "RolloutResponse": Rollout -> EmbodiedLoop  (action chunks)
"""

import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.embodied_loop.embodied_loop_worker import EmbodiedLoopWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.stateless_rollout_worker import (
        StatelessRolloutWorker,
    )


class EmbodiedLoopEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        embodied_loop: "EmbodiedLoopWorker",
        rollout: "StatelessRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.embodied_loop = embodied_loop
        self.rollout = rollout
        self.env = env

        # Env-side channels.
        self.env_action_channel = Channel.create("EnvAction")
        self.env_obs_channel = Channel.create("EnvObs")
        # Rollout-side channels.
        self.rollout_req_channel = Channel.create("RolloutRequest")
        self.rollout_resp_channel = Channel.create("RolloutResponse")

        self.run_timer = run_timer
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)
        self.logger = get_logger()

    def init_workers(self):
        loop_handle = self.embodied_loop.init_worker()
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        loop_handle.wait()
        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        # EnvWorker reads actions from `input_channel` (EnvAction) and writes
        # obs to `rollout_channel` (EnvObs). The parameter name `rollout_channel`
        # is the EnvWorker API and is kept for backward compatibility.
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_action_channel,
            rollout_channel=self.env_obs_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate_start(
            request_channel=self.rollout_req_channel,
            response_channel=self.rollout_resp_channel,
            mode="eval",
        )
        loop_handle: Handle = self.embodied_loop.evaluate(
            env_input_channel=self.env_obs_channel,
            env_output_channel=self.env_action_channel,
            rollout_req_channel=self.rollout_req_channel,
            rollout_resp_channel=self.rollout_resp_channel,
        )

        loop_handle.wait()
        env_results = env_handle.wait()
        self.rollout.evaluate_stop(request_channel=self.rollout_req_channel).wait()
        rollout_handle.wait()

        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
