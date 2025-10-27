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

import logging
import typing
from typing import Optional, Union

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.agent_loop.agent_loop import ToolAgentLoopWorker
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.mcp.tool_worker import ToolWorker
from rlinf.workers.reward.reward_worker import RewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logging.getLogger().setLevel(logging.INFO)


class ToolAgentRunner(ReasoningRunner):
    """Runner for agent task RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        reward: RewardWorker,
        agent_loop: ToolAgentLoopWorker,
        tool_workers: dict[ToolWorker, dict] = [],
    ):
        super().__init__(
            cfg,
            placement,
            train_dataset,
            val_dataset,
            rollout,
            inference,
            actor,
            reward,
        )

        self.agent_loop = agent_loop
        self.tool_workers = tool_workers
        self.generate_input_channel = Channel.create("GenerateInput")
        self.generate_output_channel = Channel.create("GenerateOutput")
        self.tool_input_channels = [Channel.create(f"Tool-{worker.worker_group_name}") for worker in self.tool_workers]
        self.tool_output_channel = Channel.create("ToolOutput")

    def init_workers(self):
        for worker, input_channel in zip(self.tool_workers, self.tool_input_channels):
            worker.init_worker(
                input_channel, self.tool_output_channel
            ).wait()
        tool_has_session = []
        tool_input_channel_dict = {}
        for (worker, worker_info), input_channel in zip(self.tool_workers.items(), self.tool_input_channels):
            tool_names = worker_info['tool_names']
            has_session = worker_info['has_session']
            if isinstance(tool_names, list):
                if has_session:
                    tool_has_session.extend(tool_names)
                for tool_name in tool_names:
                    tool_input_channel_dict[tool_name] = input_channel
            elif isinstance(tool_names, str):
                if has_session:
                    tool_has_session.append(tool_names)
                tool_input_channel_dict[tool_names] = input_channel

        self.agent_loop.init_worker(
            self.generate_input_channel,
            self.generate_output_channel,
            tool_input_channel_dict,
            self.tool_output_channel,
            tool_has_session,
        ).wait()

        super().init_workers()

    def run(self):
        epoch_iter = range(self.epoch, self.cfg.runner.max_epochs)
        if len(epoch_iter) <= 0:
            # epoch done
            return

        global_pbar = tqdm(
            initial=self.global_steps,
            total=self.max_steps,
            desc="Global Step",
            ncols=620,
        )

        self.run_timer.start_time()
        self.rollout.rollout_serverless(
            self.generate_input_channel, self.generate_output_channel
        )
        for tool_worker in self.tool_workers:
            tool_worker.start_server()
        for _ in epoch_iter:
            for batch in self.train_dataloader:
                with self.timer("step"):
                    with self.timer("prepare_data"):
                        self._put_batch(batch)

                    with self.timer("sync_weights"):
                        self._sync_weights()

                    # Rollout
                    rollout_handle: Handle = self.agent_loop.run_agentloop_rollout(
                        input_channel=self.dataloader_channel,
                        output_channel=self.rollout_channel,
                    )

                    # Rewards
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.rollout_channel,
                        output_channel=self.reward_channel,
                    )

                    if self.recompute_logprobs:
                        # Inference prev/ref logprobs
                        infer_handle: Handle = self.inference.run_inference(
                            input_channel=self.reward_channel,
                            output_channel=self.inference_channel,
                            compute_ref_logprobs=self.compute_ref_logprobs,
                        )
                        inference_channel = self.inference_channel
                    else:
                        infer_handle = None
                        inference_channel = self.reward_channel

                    # Advantages and returns
                    adv_handle: Handle = self.actor.compute_advantages_and_returns(
                        input_channel=inference_channel,
                        output_channel=self.actor_channel,
                    )

                    # Actor training
                    actor_handle: Handle = self.actor.run_training(
                        input_channel=self.actor_channel,
                    )

                    metrics = actor_handle.wait()
                    actor_rollout_metrics = metrics[0][0]
                    actor_training_metrics = metrics[0][1]
                    self.global_steps += 1

                    run_time_exceeded = self.run_timer.is_finished()
                    _, save_model, is_train_end = check_progress(
                        self.global_steps,
                        self.max_steps,
                        self.cfg.runner.val_check_interval,
                        self.cfg.runner.save_interval,
                        1.0,
                        run_time_exceeded=run_time_exceeded,
                    )

                    if save_model:
                        self._save_checkpoint()

                    if is_train_end:
                        logging.info(
                            f"Step limit given by max_steps={self.max_steps} reached. Stopping run"
                        )
                        return

                    if run_time_exceeded:
                        logging.info(
                            f"Time limit given by run_timer={self.run_timer} reached. Stopping run"
                        )
                        return

                time_metrics = self.timer.consume_durations()
                time_metrics["training"] = actor_handle.consume_duration()
                time_metrics["rollout"] = rollout_handle.consume_duration()
                time_metrics["reward"] = reward_handle.consume_duration()
                time_metrics["advantage"] = adv_handle.consume_duration()
                if infer_handle is not None:
                    # Inference time should be the min time across ranks, because different DP receive the rollout results differently
                    # But at the begin of the pp schedule, there is a timer barrier
                    # This makes all DP end at the same time, while they start at differnt times, and thus only the min time is correct
                    time_metrics["inference"] = infer_handle.consume_duration(
                        reduction_type="min"
                    )

                logging_steps = (
                    self.global_steps - 1
                ) * self.cfg.algorithm.n_minibatches
                # add prefix to the metrics
                log_time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
                rollout_metrics = {
                    f"rollout/{k}": v for k, v in actor_rollout_metrics.items()
                }

                self.metric_logger.log(log_time_metrics, logging_steps)
                self.metric_logger.log(rollout_metrics, logging_steps)
                for i in range(self.cfg.algorithm.n_minibatches):
                    training_metrics = {
                        f"train/{k}": v for k, v in actor_training_metrics[i].items()
                    }
                    self.metric_logger.log(training_metrics, logging_steps + i)

                logging_metrics = {f"{k}_time": v for k, v in time_metrics.items()}

                if self.cfg.actor.get("calculate_flops", False):
                    flops_metrics = self._compute_flops_metrics(
                        time_metrics, actor_rollout_metrics
                    )
                    flops_metrics = {f"flops/{k}": v for k, v in flops_metrics.items()}
                    self.metric_logger.log(flops_metrics, logging_steps)
                    logging_metrics.update(flops_metrics)

                logging_metrics.update(actor_rollout_metrics)
                logging_metrics.update(actor_training_metrics[-1])

                global_pbar.set_postfix(logging_metrics)
                global_pbar.update(1)

        for tool_worker in self.tool_workers:
            tool_worker.stop_server()
        self.metric_logger.finish()
