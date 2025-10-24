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
from omegaconf import DictConfig

from rlinf.scheduler import Worker, Channel

from rlinf.workers.mcp.tool_worker import ToolWorker
from rlinf.data.tool_call.tool_io_struct import ToolResponse


class FakeToolWorker(ToolWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.request_processor_task = None

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        self.input_channel = input_channel
        self.output_channel = output_channel

    def start_server(self):
        loop = asyncio.get_running_loop()
        self.request_processor_task = loop.create_task(self._process_requests())

    def stop_server(self):
        # Cancel request processor task
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()


    async def _process_requests(self):
        async def generate_and_send(channel_key: str, tool_args: dict):
            result = ToolResponse(text="fake_tool_response")
            await self.output_channel.put(result, key=channel_key, async_op=True).async_wait()

        while True:
            rollout_request = await self.input_channel.get(async_op=True).async_wait()
            asyncio.create_task(generate_and_send(rollout_request['channel_key'], rollout_request['tool_args']))
