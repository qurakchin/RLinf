# Copyright (c) 2025, RLinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained BEHAVIOR-1K streaming SFT data pipeline.

Relocated out of the ``openpi_pytorch`` model package so data loading lives with
the other datasets under ``rlinf/data/datasets``. The pipeline reuses the
vendored ``openpi_pytorch`` preprocessing primitives (state extraction, image
resize/pad, quantile normalization, the PaliGemma tokenizer) but imports no
externally installed ``openpi``. The SFT worker dispatches to
:func:`build_behavior_sft_dataloader` the same way it dispatches DreamZero.
"""

from rlinf.data.datasets.behavior.behavior_pinned_loader import (
    PinnedBehaviorSftDataLoader,
    build_pinned_behavior_sft_dataloader,
)
from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
    BehaviorSftDataConfig,
    BehaviorSftDataLoader,
    build_behavior_sft_dataloader,
    collate_behavior_sft_items,
    create_behavior_sft_data_loader,
)
from rlinf.data.datasets.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.datasets.behavior.behavior_sft_transform import (
    BehaviorSftTransform,
    transform_behavior_sft_item,
)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "BehaviorSftDataset",
    "BehaviorSftTransform",
    "PinnedBehaviorSftDataLoader",
    "build_behavior_sft_dataloader",
    "build_pinned_behavior_sft_dataloader",
    "collate_behavior_sft_items",
    "create_behavior_sft_data_loader",
    "transform_behavior_sft_item",
]
