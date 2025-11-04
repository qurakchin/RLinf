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
import random
import re
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)

from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopOutput, AgentLoopWorker

class Searchr1ToolAgentLoopWorker(AgentLoopWorker):
    """Simple tool agent loop that can interact with tools."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.tool_call_start_token: str = "<search>"
        self.tool_call_end_token: str = "</search>"
        self.tool_call_regex = re.compile(r"<search>(.*?)</search>", re.DOTALL)

    async def state_less_tool_call_with_channel(
        self,
        input_channel: Channel,
        output_channel: Channel,
        tool_name: str,
        tool_args: dict,
    ) -> ToolChannelResponse:
        """state-less tool call with channel, used for demo"""
        session_id = uuid4().hex
        await input_channel.put(
            ToolChannelRequest(
                session_id=session_id,
                request_type="execute",
                tool_name=tool_name,
                tool_args=tool_args,
            ),
            async_op=True,
        ).async_wait()
        return await output_channel.get(session_id, async_op=True).async_wait()

    async def tool_call(self, tool_request: ToolRequest) -> ToolResponse:
        tool_name, tool_args = tool_request.name, tool_request.arguments
        tool_channel_info = self.tool_channel_info_map[self.tool_name_map[tool_name]]
        channel_response = await self.state_less_tool_call_with_channel(
            tool_channel_info.input_channel,
            self.tool_worker_output_channel,
            tool_name,
            tool_args,
        )

        # no failure in this demo
        assert channel_response.success
        if isinstance(channel_response.result, (list, dict)):
            result_text = json.dumps(channel_response.result)
        else:
            result_text = str(channel_response.result)
        return ToolResponse(
            text=result_text,
        )

    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []
        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        if matches:
            match = matches[-1].strip()
            function_calls.append(ToolRequest(name='search', arguments={"keyword":match}))

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_text)

        return content, function_calls

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        trace_prints = []
        response_mask = []
        for _ in range(self.cfg.tools.maxturn):
            # Generate response from LLM
            max_prompt_len = int(self.cfg.data.get("max_prompt_length", 1024))
            max_total_len = int(self.cfg.actor.model.encoder_seq_length)
            max_resp_len = max(1, max_total_len - max_prompt_len)

            generate_result = await self.generate(prompt_ids)
            response_ids = generate_result["output_ids"]
            if len(response_ids) > max_resp_len:
                response_ids = response_ids[:max_resp_len]
            response_text = self.tokenizer.decode(response_ids)

            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)  # 1 for LLM generated tokens

            # Extract tool calls from response
            _, tool_requests = await self.extract_tool_calls(response_text)
            if tool_requests == []:
                break

            # Execute tools in parallel with history propagation
            tasks = []
            for tool_request in tool_requests:
                tasks.append(self.tool_call(tool_request))
            tool_responses: list[ToolResponse] = await asyncio.gather(*tasks)

            # Convert tool responses to messages and tokenize
            tool_messages = []
            for tool_response in tool_responses:
                message = tool_response.text
                tool_messages.append(message)
            # Tokenize tool responses
            tool_response_ids = self.tokenizer.encode(tool_messages[0], add_special_tokens=False)
            prompt_ids += tool_response_ids
            response_mask += [0] * len(tool_response_ids)
            if self.print_outputs:
                # add anything you want to print
                trace_prints.append({"generate": response_text, "tool_resp": tool_messages})

        # Separate prompt and response
        response_ids = prompt_ids[len(orig_prompt_ids) :]

        return AgentLoopOutput(
            prompt_ids=orig_prompt_ids,
            prompt_text=self.tokenizer.decode(orig_prompt_ids),
            response_ids=response_ids,
            response_mask=response_mask,
            trace_prints=trace_prints,
        )
