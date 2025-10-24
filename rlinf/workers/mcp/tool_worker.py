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

from rlinf.scheduler import Worker, Channel


class ToolWorker(Worker):

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        """Initialize the worker with communication channels."""
        self.input_channel = input_channel
        self.output_channel = output_channel


    def start_server(self):
        raise NotImplementedError()


    def stop_server(self):
        raise NotImplementedError()

