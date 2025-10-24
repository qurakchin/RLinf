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

import logging
import os
import typing
from typing import Dict, Optional, Union

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from rlinf.workers.agent_loop.tool_agent_loop import ToolAgentLoopWorker
from rlinf.workers.mcp.tool_worker import ToolWorker
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.reward.reward_worker import RewardWorker
from .agent_runner import AgentRunner

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class ToolAgentRunner(AgentRunner):
    """Runner for agent task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        generate: Union["SGLangWorker", "VLLMWorker"],
        agent_loop: ToolAgentLoopWorker,
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        reward: RewardWorker,
        tool_workers: dict[str, ToolWorker]={},
    ):
        super().__init__(
            cfg,
            placement,
            train_dataset,
            val_dataset,
            generate,
            agent_loop,
            inference,
            actor,
            reward,
        )
        self.tool_workers = tool_workers
        self.tool_input_channels = {
            name: Channel.create(f"Tool-{name}") for name in tool_workers.keys()
        }
        self.tool_output_channel = Channel.create(f"ToolOutput")

    def init_workers(self):
        super().init_workers(init_agent_loop=False)
        for name, worker in self.tool_workers.items():
            worker.init_worker(self.tool_input_channels[name], self.tool_output_channel).wait()
        self.agent_loop.init_worker(self.generate_input_channel, self.generate_output_channel, self.tool_input_channels, self.tool_output_channel).wait()

    def run(self):
        for tool_worker in self.tool_workers.values():
            tool_worker.start_server()
        super().run()