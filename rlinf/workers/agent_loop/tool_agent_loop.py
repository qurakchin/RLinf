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
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4
from typing import Dict, List
from dataclasses import dataclass, field
from megatron.core import parallel_state
from rlinf.scheduler import Channel, Worker
from transformers import AutoTokenizer
from omegaconf import DictConfig
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
)

from rlinf.data.tool_call.tool_parser import HermesToolParser, FakeToolParser, ToolParser
from rlinf.data.tool_call.tool_io_struct import ToolRequest, ToolResponse
from .agent_loop import AgentLoopWorkerBase, AgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("RLINF_LOGGING_LEVEL", "WARN"))

class ToolAgentLoopWorker(AgentLoopWorkerBase):
    """Simple tool agent loop that can interact with tools.
    """

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
    ):
        super().init_worker(generate_input_channel, generate_output_channel)
        self.tool_worker_input_channels = tool_worker_input_channels
        self.tool_worker_output_channel = tool_worker_output_channel

    async def atool_call(self, tool_request: ToolRequest) -> ToolResponse:
        tool_name, tool_args = tool_request.name, tool_request.arguments
        tool_input_channel = self.tool_worker_input_channels[tool_name]
        channel_key = uuid4().hex
        await tool_input_channel.put({'channel_key': channel_key, 'tool_args': tool_args}, async_op=True).async_wait()
        return await self.tool_worker_output_channel.get(channel_key, async_op=True).async_wait()

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        for _ in range(5):
            # Generate response from LLM
            max_prompt_len = int(self.cfg.data.get("max_prompt_length", 1024))
            max_total_len = int(self.cfg.actor.model.encoder_seq_length)
            max_resp_len = max(1, max_total_len - max_prompt_len)

            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            generate_result = await self.agenerate(prompt_ids)
            response_ids = generate_result['output_ids']
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]
            response_text = self.tokenizer.decode(response_ids)

            prompt_ids += response_ids

            # Extract tool calls from response
            _, tool_requests = await self.tool_parser.extract_tool_calls(response_text)

            # Execute tools in parallel with history propagation
            tasks = []
            for tool_request in tool_requests:
                tasks.append(self.atool_call(tool_request))
            tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

            # Convert tool responses to messages and tokenize
            tool_messages = []
            for tool_response in tool_responses:
                message = {"role": "tool", "content": tool_response.text}
                tool_messages.append(message)

            # Tokenize tool responses
            tool_response_ids = self.tokenizer.apply_chat_template(
                tool_messages, add_generation_prompt=True, tokenize=True,
            )
            prompt_ids += tool_response_ids

        # Separate prompt and response
        response_ids = prompt_ids[-len(orig_prompt_ids):]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
        )
