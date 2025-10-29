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

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import AutoTokenizer

from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
)
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLINF_LOGGING_LEVEL", "WARN"))


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    """Prompt token ids."""
    prompt_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_ids: list[int]
    """Prompt text decoded from prompt_ids"""
    prompt_text: str = ""
    """Response text decoded from response_ids"""
    response_text: str = ""
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_mask: list[int] = None
    """Log probabilities for the response tokens."""
    response_logprobs: Optional[list[float]] = None
    """Number of chat turns, including user, assistant, tool."""
    num_turns: int = 0
    """Extra fields for dynamic addition."""
    extra_fields: dict[str, Any] = {}


class AgentLoopWorkerBase(Worker):
    """Simple tool agent loop that can interact with tools.

    重构为普通类：不再继承 Worker，也不使用 WorkerGroup/Channel。
    通过注入的 rollout 直接调用 agenerate，输入/输出均为 token ids。
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rollout.model_dir)

    def init_worker(
        self,
        generate_input_channel,
        generate_output_channel,
        tool_channel_info,
        tool_name_map,
        tool_worker_output_channel: Channel,
    ):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel
        self.tool_channel_info = tool_channel_info
        self.tool_name_map = tool_name_map
        self.tool_worker_output_channel = tool_worker_output_channel

    async def agenerate(self, prompt_ids: list[int]):
        channel_key = uuid4().hex
        await self.generate_input_channel.put(
            {"channel_key": channel_key, "prompt_ids": prompt_ids}, async_op=True
        ).async_wait()
        result = await self.generate_output_channel.get(
            channel_key, async_op=True
        ).async_wait()
        return result

    async def run_agentloop_rollout(
        self, input_channel: Channel, output_channel: Channel
    ):
        """
        Run the agent loop for multiple queries.
        """
        with self.worker_timer():
            rollout_request: RolloutRequest = input_channel.get()

            send_output_tasks = []
            for input_ids, answers in zip(
                rollout_request.input_ids, rollout_request.answers
            ):
                rollout_tasks = []
                # grpo group_size
                for _ in range(rollout_request.n):
                    task = asyncio.create_task(self.run_one_query(input_ids))
                    rollout_tasks.append(task)

                task_results = await asyncio.gather(*rollout_tasks)

                rollout_result = self._get_rollout_result(task_results, answers)

                send_output_tasks.append(
                    output_channel.put(rollout_result, async_op=True).async_wait()
                )

            await asyncio.gather(*send_output_tasks)

    def _get_rollout_result(
        self, task_results: list[AgentLoopOutput], answers
    ) -> RolloutResult:
        # Clip to model limits to avoid mask/position size mismatch
        max_prompt_len = int(self.cfg.data.max_prompt_length)
        max_total_len = int(self.cfg.actor.model.encoder_seq_length)
        max_resp_len = max(1, max_total_len - max_prompt_len)

        prompt_ids = [r.prompt_ids[:max_prompt_len] for r in task_results]
        response_ids = [r.response_ids[:max_resp_len] for r in task_results]
        prompt_lengths = [len(p) for p in prompt_ids]
        response_lengths = [len(o) for o in response_ids]
        # response_mask = [r.response_mask[:max_resp_len] for r in task_results]
        is_end = [True for _ in task_results]
        answers = [answers] * len(task_results)
        return RolloutResult(
            num_sequence=len(task_results),
            group_size=len(task_results),
            prompt_lengths=prompt_lengths,
            prompt_ids=prompt_ids,
            response_lengths=response_lengths,
            response_ids=response_ids,
            is_end=is_end,
            answers=answers,
            # response_mask=response_mask,
        )

    async def run_one_query(self, prompt_ids: list[int], **kwargs) -> AgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")
