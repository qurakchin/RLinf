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
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import (
    ToolChannelRequest,
    ToolChannelResponse,
    ToolRequest,
    ToolResponse,
)
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopOutput, AgentLoopWorker
from multiturn_demo.tools.tool_parser import HermesToolParser


@dataclass
class GenerateContext:
    tool_session_ids: dict[str, str] = field(default_factory=dict)


class HttpAgentLoopWorker(AgentLoopWorker):
    """
    An agent loop worker that can interact with http tools with session.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        self.tool_parser = HermesToolParser()

        self.tool_response_sys_prefix_len = len(self.tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True))

    def generate_context_create(self) -> dict[str, Any]:
        return GenerateContext()

    async def generate_context_release(
        self, generate_context: GenerateContext
    ) -> dict[str, Any]:
        for tool_worker_name, session_id in generate_context.tool_session_ids.items():
            if self.tool_channel_info_map[tool_worker_name].has_session:
                # tool need session
                await self.tool_session_release(tool_worker_name, session_id)

    async def tool_session_get(
        self, generate_context: GenerateContext, tool_name: str
    ) -> Any:
        tool_worker_name = self.tool_name_map[tool_name]
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        if tool_worker_name in generate_context.tool_session_ids:
            return generate_context.tool_session_ids[tool_worker_name]
        session_id = uuid4().hex
        generate_context.tool_session_ids[tool_worker_name] = session_id
        if tool_channel_info.has_session:
            # tool need session
            await tool_channel_info.input_channel.put(
                ToolChannelRequest(session_id=session_id, request_type="session_start"),
                async_op=True,
            ).async_wait()
            self.log_debug("session_start put")
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            self.log_debug("session_start get")
        return session_id

    async def tool_session_release(self, tool_worker_name, session_id) -> str | dict:
        tool_channel_info = self.tool_channel_info_map[tool_worker_name]
        await tool_channel_info.input_channel.put(
            ToolChannelRequest(session_id=session_id, request_type="session_end"),
            async_op=True,
        ).async_wait()
        self.log_debug("session_end put")
        response: ToolChannelResponse = await self.tool_worker_output_channel.get(
            session_id, async_op=True
        ).async_wait()
        self.log_debug("session_end get")

    async def tool_call(
        self, generate_context: GenerateContext, tool_request: ToolRequest | ToolChannelResponse
    ) -> ToolResponse:
        if isinstance(tool_request, ToolChannelResponse):
            return ToolResponse(text=tool_request.result)
        elif tool_request.name not in self.tool_name_map:
            return ToolResponse(text=f"Error: tool {tool_request.name} not exist")
        else:
            tool_name, tool_args = tool_request.name, tool_request.arguments
            tool_channel_info = self.tool_channel_info_map[self.tool_name_map[tool_name]]
            tool_input_channel = tool_channel_info.input_channel
            session_id = await self.tool_session_get(generate_context, tool_name)
            await tool_input_channel.put(
                ToolChannelRequest(
                    session_id=session_id,
                    request_type="execute",
                    tool_name=tool_name,
                    tool_args=tool_args,
                ),
                async_op=True,
            ).async_wait()
            self.log_debug("tool execute put")
            response: ToolChannelResponse = await self.tool_worker_output_channel.get(
                session_id, async_op=True
            ).async_wait()
            self.log_debug("tool execute get")
            self.log_info(f"{response}")
            if isinstance(response.result, (list, dict)):
                result_text = json.dumps(response.result)
            else:
                result_text = str(response.result)
            return ToolResponse(text=result_text)

    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest | ToolChannelResponse]]:
        _, toolcalls = await self.tool_parser.extract_tool_calls(response_text)
        return_function_calls = []
        for toolcall in toolcalls:
            if isinstance(toolcall, ToolChannelResponse):
                return_function_calls.append(toolcall)
            else:
                try:
                    arguments = json.loads(toolcall.arguments)
                    return_function_calls.append(ToolRequest(name=toolcall.name, arguments=arguments))
                except Exception as e:
                    return_function_calls.append(
                        ToolChannelResponse(success=False, result=f"Error: Failed to decode tool call arguments: {e}")
                    )

        return response_text, return_function_calls

    async def run_one_query(self, prompt_ids: list[int]) -> AgentLoopOutput:
        max_prompt_len = int(self.cfg.data.get("max_prompt_length", 1024))
        prompt_ids = prompt_ids[:max_prompt_len]
        orig_prompt_ids = copy.deepcopy(prompt_ids)
        generate_context: GenerateContext = self.generate_context_create()
        trace_prints = []
        response_mask = []
        max_total_len = int(self.cfg.actor.model.encoder_seq_length) - max_prompt_len + len(orig_prompt_ids)
        try:
            # 5 is a magic number in this demo.
            for _ in range(5):
                # Generate response from LLM
                generate_result = await self.generate(prompt_ids)
                response_ids = generate_result["output_ids"]
                prompt_ids += response_ids
                response_mask += [1] * len(response_ids)  # 1 for LLM generated tokens
                print(f"[{len(orig_prompt_ids)}] {len(prompt_ids)} {len(response_ids)} {len(response_mask)}")

                if len(prompt_ids) >= max_total_len:
                    prompt_ids = prompt_ids[:max_total_len]
                    break

                response_text = self.tokenizer.decode(response_ids)
                print(f"{response_text=}")
                # Extract tool calls from response
                _, tool_requests = await self.extract_tool_calls(response_text)

                if not tool_requests:
                    break

                self.log_info(f"{tool_requests}")

                # Execute tools in parallel with history propagation
                tasks = []
                for tool_request in tool_requests:
                    tasks.append(self.tool_call(generate_context, tool_request))
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
                )[self.tool_response_sys_prefix_len:]
                
                if len(prompt_ids) + len(tool_response_ids) >= max_total_len:
                    break

                prompt_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)  # 0 for tool response tokens
                print(f"[{len(orig_prompt_ids)}] {len(prompt_ids)} {len(response_ids)} {len(response_mask)}")

                if self.print_outputs:
                    # add anything you want to print
                    trace_prints.append(
                        {"generate": response_text, "tool_resp": tool_messages}
                    )

            # Separate prompt and response
            print(f"{len(prompt_ids)}({len(orig_prompt_ids)}+?) {max_total_len} {len(response_mask)}")
            response_ids = prompt_ids[len(orig_prompt_ids):]
            response_mask = response_mask[:max_total_len - len(orig_prompt_ids)]

            assert len(response_mask) == len(response_ids), f"{len(response_mask)} != {len(response_ids)}"

            return AgentLoopOutput(
                prompt_ids=orig_prompt_ids,
                prompt_text=self.tokenizer.decode(orig_prompt_ids),
                response_text=self.tokenizer.decode(response_ids),
                response_ids=response_ids,
                response_mask=response_mask,
                trace_prints=trace_prints,
            )
        finally:
            await self.generate_context_release(generate_context)
