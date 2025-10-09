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


from rlinf.algorithms.registry import get_reward_fn


def register_rewards():
    try:
        from toolkits.code_verifier.verify import fim_verify_call
        assert get_reward_fn("fim_verify_call") == fim_verify_call
    except ImportError:
        pass

    try:
        from toolkits.math_verifier.verify import math_verify_call
        assert get_reward_fn("math") == math_verify_call
    except ImportError:
        pass
