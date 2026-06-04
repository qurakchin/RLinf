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

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_loop_eval_runner import EmbodiedLoopEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.embodied_loop.embodied_loop_worker import EmbodiedLoopWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.stateless_rollout_worker import StatelessRolloutWorker

mp.set_start_method("spawn", force=True)


"""Eval entry for the EmbodiedLoop + StatelessRollout split.

Launches three groups:
    - env_group      (EnvWorker)                          -> placement key "env"
    - rollout_group  (StatelessRolloutWorker, stateless)  -> placement key "rollout"
                       (kept so that EnvWorker, which hard-codes "rollout" for
                       its peer world_size, sees the right counterpart)
    - loop_group     (EmbodiedLoopWorker, stateful)       -> placement key "embodied_loop"
"""

@hydra.main(
    version_base="1.1", config_path="config", config_name="libero_spatial_ppo_openpi_pi05_embloop",
)
def main(cfg) -> None:
    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # EmbodiedLoop group (placement key "embodied_loop").
    loop_placement = component_placement.get_strategy("embodied_loop")
    loop_group = EmbodiedLoopWorker.create_group(cfg).launch(
        cluster,
        name=cfg.embodied_loop.group_name,
        placement_strategy=loop_placement,
    )

    # Create stateless rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = StatelessRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedLoopEvalRunner(
        cfg=cfg,
        embodied_loop=loop_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
