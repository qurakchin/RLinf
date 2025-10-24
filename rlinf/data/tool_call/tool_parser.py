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

import json
import logging
import random
import regex as re
from .tool_io_struct import ToolRequest, ToolResponse


class ToolParser:

    def __init__(self, cfg, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.logger = logger

    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        """Extract tool calls from the responses.

        Args:
            responses_ids (List[int]): The ids of the responses.

        Returns:
            Tuple[str, List[ToolRequest]]: Remaining content and extracted tool calls.
        """
        raise NotImplementedError


class HermesToolParser(ToolParser):
    """Adapted from https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py"""

    def __init__(self, cfg, logger) -> None:
        super().__init__(cfg, logger)

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

        self.keep_match_error_as_response = True
        # self.keep_match_error = cfg.get("keep_match_error", False)
        self.keep_match_error_as_response_text = "Failed to decode tool call: {exception}"


    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        if self.tool_call_start_token not in response_text or self.tool_call_end_token not in response_text:
            return response_text, []

        matches = self.tool_call_regex.findall(response_text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(ToolRequest(name=name, arguments=arguments))
            except Exception as exception:
                self.logger.error(f"Failed to decode tool call: {exception}")
                if self.keep_match_error_as_response:
                    function_calls.append(ToolResponse(text=self.keep_match_error_as_response_text.format(exception=exception)))

        # remaining text exclude tool call tokens
        content = self.tool_call_regex.sub("", response_text)

        return content, function_calls


class FakeToolParser(ToolParser):
    def __init__(self, cfg, logger) -> None:
        super().__init__(cfg, logger)

    async def extract_tool_calls(self, response_text) -> tuple[str, list[ToolRequest]]:
        function_calls = [
            [ToolRequest(name='tool1', arguments={'arg1': 'value1'})],
            [ToolRequest(name='tool1', arguments={'arg2': 'value2'})],
        ]
        return_function_calls = random.choice(function_calls)

        return response_text, return_function_calls
