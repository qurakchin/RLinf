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

import json
import logging
import re

import regex

from rlinf.algorithms.registry import register_toolcall_parser
from rlinf.data.tool_call.tool_io_struct import (
    ToolRequest,
    ToolResponse,
)


@register_toolcall_parser("qwen2.5")
class Qwen25ToolCallParser:
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self):
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = regex.compile(
            r"<tool_call>(.*?)</tool_call>", regex.DOTALL
        )

    async def __call__(self, responses_text: str) -> tuple[str, list[ToolRequest]]:
        text = responses_text
        if (
            self.tool_call_start_token not in text
            or self.tool_call_end_token not in text
        ):
            return text, []

        matches = self.tool_call_regex.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(
                    ToolRequest(
                        name=name, arguments=json.dumps(arguments, ensure_ascii=False)
                    )
                )
            except Exception as e:
                logging.error(f"Failed to decode tool call: {e}")

        # remaing text exclude tool call tokens
        content = self.tool_call_regex.sub("", text)

        return content, function_calls


@register_toolcall_parser("searchr1-qwen2.5")
class Searchr1Qwen25ToolCallParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<search>"
        self.tool_call_end_token: str = "</search>"
        self.tool_call_regex = re.compile(r"<search>(.*?)</search>", re.DOTALL)

    async def __call__(self, response_text: str) -> tuple[str, list[ToolRequest]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []
        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        if matches:
            match = matches[-1].strip()
            function_calls.append(
                ToolRequest(name="search", arguments={"keyword": match})
            )

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_text)

        return content, function_calls


@register_toolcall_parser("rstar2-qwen")
class Rstar2QwenToolCallParser:
    def __init__(self) -> None:
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    async def __call__(
        self, response_text: str
    ) -> tuple[str, list[ToolRequest | ToolResponse]]:
        if (
            self.tool_call_start_token not in response_text
            or self.tool_call_end_token not in response_text
        ):
            return response_text, []

        matches = self.tool_call_regex.findall(response_text)
        return_function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                return_function_calls.append(
                    ToolRequest(name=name, arguments=arguments)
                )
            except Exception as e:
                return_function_calls.append(
                    ToolResponse(text=f"Failed to decode tool call: {e}")
                )

        return response_text, return_function_calls
