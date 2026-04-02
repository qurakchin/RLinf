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

from __future__ import annotations

from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from agentlightning.algorithm.base import Algorithm
from agentlightning.types import Dataset

from .entrypoint import run_rlinf_training


class RlinfAlgorithm(Algorithm):
    """Add rlinf tag to agentlightning algorithm."""

    def __init__(
        self,
        config: dict[str, Any] | DictConfig,
        eval: bool = False,
    ):
        super().__init__()

        if isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            self.config = config
        self.eval = eval

    def run(
        self,
        train_dataset: Optional[Dataset[Any]] = None,
        val_dataset: Optional[Dataset[Any]] = None,
    ) -> None:
        store = self.get_store()
        adapter = self.get_adapter()
        run_rlinf_training(
            config=self.config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            store=store,
            adapter=adapter,
            eval=self.eval,
        )
