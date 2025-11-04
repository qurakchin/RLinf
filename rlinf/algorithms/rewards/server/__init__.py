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

import re
import requests

from typing import List, Tuple, Optional
from omegaconf import DictConfig

from toolkits.math_verifier.verify import math_verify_call


class ServerReward:
    def __init__(self, config: DictConfig):
        self.scale = config.get("reward_scale", 1.0)

    def get_reward(
        self, response: List[str], reference: List[List[str]]
    ) -> List[float]:
        """
        Calculates reward scores for a list of responses compared to corresponding lists of reference answers.
        For each response, the function checks if it matches any of the provided references using the `process_results` function.
        The reward for each response is computed as the first element of the result (converted to float) multiplied by `self.scale`.
        Args:
            response (List[str]): A list of response strings to be evaluated.
            reference (List[List[str]]): A list where each element is a list of reference strings corresponding to each response.
        Returns:
            List[float]: A list of reward scores, one for each response.
        """
        for res in response:
            print(f"res={repr(res)}")
        code_snippets = [extract_code_solution(res)[0] for res in response]
        rewards = [0.] * len(code_snippets)
        print(f"{self.__class__.__name__}.get_reward: len(rewards)={len(rewards)}")
        code_snippets = [(idx, cs) for idx, cs in enumerate(code_snippets) if cs is not None]

        if len(code_snippets) == 0:
            return rewards

        code_idxs, code_snippets = tuple(zip(*code_snippets))

        submissions = [
            {
                "type": "lean",
                "solution": code_snippet,
            } for code_snippet in code_snippets
        ]
        print(f"{self.__class__.__name__}.get_reward: len(submissions)={len(submissions)}")

        data = {
            "type": "batch",
            "submissions": submissions
        }

        response = requests.post(
            "http://127.0.0.1:8088//run/long-batch",
            json=data
        )
        results = response.json()["results"]
        results = [float(result["run_success"] and result["success"]) for result in results]

        for idx, result in zip(code_idxs, results):
            rewards[idx] = result
        print(f"{self.__class__.__name__}.get_reward: {rewards=}")

        return [
            float(1 if is_correct else -1) * self.scale
            for is_correct in rewards
        ]


def extract_code_solution(solution_str: str) -> Tuple[Optional[str], str]:
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        # processed_str = solution_str.split("Assistant:", 1)[1]
        # For multi-turn response, the last assistant response is the final answer
        processed_str = solution_str.rsplit("Assistant:", 1)[1]
        question_str = solution_str.split("Assistant:", 1)[0]
    elif "<|im_start|>assistant" in solution_str:
        # processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
        # For multi-turn response, the last assistant response is the final answer
        processed_str = solution_str.rsplit("<|im_start|>assistant", 1)[1]
        question_str = solution_str.split("<|im_start|>assistant", 1)[0]
    else:
        # print("[Error] Failed to locate model response header")
        # return None, solution_str, ""
        processed_str = solution_str
        question_str = None

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        return None, processed_str, question_str

    final_answer = matches[-1].group(1).strip()

    final_answer = processed_str
    if '```python' in final_answer or '```cpp' in final_answer or '```lean' in final_answer:
        final_answer = extract_program_in_delimiter(final_answer, last_only=True)
    return final_answer, processed_str, question_str


def extract_program_in_delimiter(result: str, last_only=False):
    program = ""
    start = False
    for line in result.split("\n"):
        if line.find("```python") != -1 or line.find("```cpp") != -1 or line.find("```lean") != -1:
            if last_only:
                program = "" # only extract the last program
            else:
                program += "\n# ========\n"
            start = True
        elif line.find("```") != -1:
            start = False
        elif start:
            program += line + "\n"
    # maybe all output is a program
    if not program:
        return result
    return program.strip()
