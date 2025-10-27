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
import copy
import logging
import os
from typing import Any, Optional
from uuid import uuid4
import json

from omegaconf import DictConfig
from pydantic import BaseModel
from transformers import AutoTokenizer

from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
)
from rlinf.data.tool_call.tool_io_struct import ToolRequest, ToolResponse
from rlinf.data.tool_call.tool_parser import FakeToolParser, ToolParser
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

    def init_worker(self, generate_input_channel, generate_output_channel):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel

    async def agenerate(self, prompt_ids: list[int]):
        session_id = uuid4().hex
        await self.generate_input_channel.put(
            {"session_id": session_id, "prompt_ids": prompt_ids}, async_op=True
        ).async_wait()
        result = await self.generate_output_channel.get(
            session_id, async_op=True
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
            for input_ids, answers in zip(rollout_request.input_ids, rollout_request.answers):
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

    def generate_context_create(self) -> dict[str, Any]:
        return {
            'generate_instance_id': None,
        }

    async def generate_context_release(self, generate_context) -> dict[str, Any]:
        pass

    async def run_one_query(self, prompt_ids: list[int], **kwargs) -> AgentLoopOutput:
        raise NotImplementedError("Subclasses must implement this method")


class ToolAgentLoopWorker(AgentLoopWorkerBase):
    """Simple tool agent loop that can interact with tools."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)

        # Configuration

        # Initialize tool parser
        self.tool_parser: ToolParser = FakeToolParser(cfg, self._logger)

    def init_worker(
        self,
        generate_input_channel,
        generate_output_channel,
        tool_worker_input_channels: dict[str, Channel],
        tool_worker_output_channel: Channel,
        tool_has_session: list[str]=[],
    ):
        # print(f"zcy_dbg: ToolAgentLoopWorker.init_worker: tool_worker_input_channels.keys(): {tool_worker_input_channels.keys()}")
        # exit(-1)
        super().init_worker(generate_input_channel, generate_output_channel)
        self.tool_worker_input_channels = tool_worker_input_channels
        self.tool_worker_output_channel = tool_worker_output_channel
        self.tool_has_session = tool_has_session

    def generate_context_create(self) -> dict[str, Any]:
        return {
            **super().generate_context_create(),
            'tool_session_ids': {},
        }

    async def generate_context_release(self, generate_context) -> dict[str, Any]:
        for tool_name, session_id in generate_context['tool_session_ids'].items():
            if tool_name in self.tool_has_session:
                # tool need session
                await self.tool_session_release(tool_name, session_id)
        return await super().generate_context_release(generate_context)

    async def tool_session_get(self, generate_context, tool_name: str) -> Any:
        if tool_name in generate_context['tool_session_ids']:
            return generate_context['tool_session_ids'][tool_name]
        session_id = uuid4().hex
        generate_context['tool_session_ids'][tool_name] = session_id
        if tool_name in self.tool_has_session:
            # tool need session
            await self.tool_worker_input_channels[tool_name].put(
                {"session_id": session_id, "request_type": "session_start"}, async_op=True
            ).async_wait()
            response_dict = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            assert response_dict['success']
        return session_id

    async def tool_session_release(self, tool_name, session_id) -> str | dict:
        await self.tool_worker_input_channels[tool_name].put(
            {"session_id": session_id, "request_type": "session_end"}, async_op=True
        ).async_wait()
        response_dict = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        assert response_dict['success']

    async def atool_call(self, generate_context, tool_request: ToolRequest) -> ToolResponse:
        tool_name, tool_args = tool_request.name, tool_request.arguments
        tool_input_channel = self.tool_worker_input_channels[tool_name]
        session_id = await self.tool_session_get(generate_context, tool_name)
        await tool_input_channel.put(
            {"session_id": session_id, "request_type": "session_execute", "tool_name": tool_name, "tool_args": tool_args}, async_op=True
        ).async_wait()
        response_dict = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        assert response_dict['success']
        if isinstance(response_dict['result'], (list, dict)):
            result_text = json.dumps(response_dict['result'])
        else:
            result_text = str(response_dict['result'])
        return ToolResponse(
            text=result_text,
        )

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        generate_context: dict[str, Any] = self.generate_context_create()
        try:
            for _ in range(5):
                # Generate response from LLM
                max_prompt_len = int(self.cfg.data.get("max_prompt_length", 1024))
                max_total_len = int(self.cfg.actor.model.encoder_seq_length)
                max_resp_len = max(1, max_total_len - max_prompt_len)

                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

                generate_result = await self.agenerate(prompt_ids)
                response_ids = generate_result["output_ids"]
                if len(response_ids) > max_resp_len:
                    response_ids = response_ids[:max_resp_len]
                response_text = self.tokenizer.decode(response_ids)

                prompt_ids += response_ids

                # Extract tool calls from response
                _, tool_requests = await self.tool_parser.extract_tool_calls(response_text)

                # Execute tools in parallel with history propagation
                tasks = []
                for tool_request in tool_requests:
                    tasks.append(self.atool_call(generate_context, tool_request))
                tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

                # Convert tool responses to messages and tokenize
                tool_messages = []
                for tool_response in tool_responses:
                    message = {"role": "tool", "content": tool_response.text}
                    tool_messages.append(message)

                # Tokenize tool responses
                tool_response_ids = self.tokenizer.apply_chat_template(
                    tool_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                prompt_ids += tool_response_ids

            # Separate prompt and response
            response_ids = prompt_ids[-len(orig_prompt_ids) :]

            return AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
            )
        finally:
            await self.generate_context_release(generate_context)
