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

from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker
from rlinf.scheduler.collective.async_channel_iter import AsyncChannelIter


class FakeToolWorker(ToolWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        super().init_worker(input_channel, output_channel)
        self.input_channel_iter: AsyncChannelIter[ToolChannelRequest] = AsyncChannelIter(input_channel)

    async def generate_and_send(self, session_id: str, tool_args: dict):
        response = ToolChannelResponse(
            success=True,
            result="fake_tool_response",
        )
        await self.output_channel.put(
            response, key=session_id, async_op=True
        ).async_wait()
        self.logger.info("FakeToolWorker._process_requests: sent response")

    async def start_server(self):
        self.input_channel_iter.set_exit_flag(False)
        output_handles = []
        async for request in self.input_channel_iter:
            self.logger.info("FakeToolWorker._process_requests: got request")
            assert request.request_type == "execute"
            assert request.tool_name == "fake_tool"
            output_handles.append(asyncio.create_task(
                self.generate_and_send(request.session_id, request.tool_args)
            ))
        await asyncio.gather(*output_handles)

    async def stop_server(self):
        # wait 0.1s to ensure start_server is called
        await asyncio.sleep(0.1)
        self.input_channel_iter.set_exit_flag()
