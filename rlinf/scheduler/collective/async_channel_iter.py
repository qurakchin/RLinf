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
from typing import Generic, TypeVar
from rlinf.scheduler.channel.channel import DEFAULT_KEY, Channel

T = TypeVar('T')
class AsyncChannelIter(Generic[T]):
    """
    used to wrap a channel and use for loop over the channel. so that the loop can be exited when the channel is empty and the exit flag is set.
    example:
        async_channel_iter = AsyncChannelIter(channel)

        # get items in a async task
        async for item in async_channel_iter:
            print(item)

        # set exit flag
        async_channel_iter.set_exit_flag()
    """
    def __init__(self, channel: Channel, key=DEFAULT_KEY):
        self.channel = channel
        self.key = key
        self.exit_flag = False
        self.unique_object = object()

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        f = self.unique_object
        while f is self.unique_object:
            if self.exit_flag and self.channel.qsize() == 0:
                raise StopAsyncIteration
            try:
                f = self.channel.get_nowait(key=self.key)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
        return f

    def set_exit_flag(self, flag=True):
        self.exit_flag = flag
