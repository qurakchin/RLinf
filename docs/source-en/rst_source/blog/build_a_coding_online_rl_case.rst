Build an Online RL Code Completion Case
=============================================

Last Updated: 10/20/2025.

Related reading: :doc:`Online RL for Code Completion <../examples/coding_online_rl>`.

1. Overview
------


In search and recommendation scenarios, Online Learning has already proven effective. For intelligent code completion in code editors, we ask: can we use real user “accept/reject” signals as immediate feedback to continuously optimize online performance via Online RL? Inspired by the Cursor team’s practice (see `blog <https://mp.weixin.qq.com/s/ShalRibfp9YSE5UFS0GLVg>`_), we explore improving completion quality and acceptance rate under real interactions.

Why is this problem suitable for Online RL:
1. The interaction is a natural closed loop: the model proposes the next action and user behavior directly forms the reward feedback.
2. Requirements and scenarios evolve continuously, making it difficult to rely on the long cycle of “build offline dataset -> infrequent offline training”.
3. There is sufficient, measurable real-time user feedback that can serve as both an online metric and a training signal.

Based on the above, we implemented a working prototype system on the RLinf framework (hereafter “prototype system”, functionally the same as a typical “demo”), focusing on the code completion task:
- Integrate data collection on the VSCode side (based on Continue), reporting prompt and accepted flag when users press Tab or abandon.
- Close the loop of online inference/training in RLinf to ensure service and training can run in parallel without blocking each other.
- Apply a series of light algorithmic and engineering optimizations tailored to the completion task.

The following sections introduce: task and data collection, system implementation (based on RLinf), algorithm and offline validation, and finally conclusions and value judgments. Note that RLinf here serves as a distributed runtime and high-performance communication substrate, enabling us to orchestrate inference and training in a single framework.

2. Code Completion Task
-------------

For editors like Cursor, one core capability is efficient code completion. For the completion task, the basic functionality is: when the cursor is at a position, the assistant provides suggested inserted content. This can be modeled as a FIM (Fill-In-the-Middle) task: given the prefix and suffix, the LLM returns the middle completion. Most modern LLMs have been pre-trained with FIM. By formatting the model input as `<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>`, the model can produce the completion.

We built the prototype’s data collection using the open-source AI coding plugin Continue. Continue already supports providing code completion via the FIM task format but does not support uploading human feedback rewards. We made a small modification: when the user presses Tab, the plugin reports the prompt with accepted=True; when the user does not press Tab for over 10 seconds or performs other code operations, it reports the prompt with accepted=False. This way we directly obtain human feedback, which can be used both as a real-time metric and as the training signal, without training a separate reward model.

In actual Cursor usage, beyond inline completion, there are advanced features like multi-line completion and applying recent edits to a similar code span. Unfortunately, such “keep-pressing-Tab” features are beyond the FIM task’s expressiveness and require more advanced task types.

3. System Implementation
----------

To support the goal of “serving users while training”, we adopt a “decoupled inference and training” deployment in RLinf and use `Worker`/`WorkerGroup` as component abstraction and orchestration. The RLinf framework provides the Worker programming interface, the foundation for building the whole system. Specifically:

- Worker represents a remote process or compute unit. By subclassing Worker, one can abstract execution logic, interact with other Workers, and be managed by RLinf.
- WorkerGroup is a helper that creates and manages a set of homogeneous Workers. By calling Worker.create_group().launch(cluster, placement), we can spin up a group of Worker instances on cluster resources, and call them remotely via the WorkerGroup interface.

The main components in the math RL training code are:

- Runner: the orchestrator that coordinates RolloutWorker / InferenceWorker / ActorWorker and others to run successfully
- RolloutWorker / InferenceWorker / ActorWorker: abstractions of rollout actors in RL training. Inference computes pref_logprobs to avoid the situation where Actor updates block computing pref_logprobs on subsequent data

We adapt the existing math RL training code to Online RL. The key framework-level adjustments are:

- Due to the need to serve online users, a shared scheduling strategy would block completion during training. We must use a decoupled scheduling strategy.
- Keep RolloutWorker / InferenceWorker / ActorWorker, but replace the data flow RolloutWorker -> InferenceWorker -> ActorWorker with ServerRolloutWorker -> InferenceWorker -> ActorWorker by introducing a ServerRolloutWorker that receives data reported by users.
- Since a static training dataset is no longer needed, RLRunner drops dataset-related logic and introduces an OnlineRouterWorker to handle online requests, using OnlineRouterWorker -> RolloutWorker to serve traffic. Meanwhile, we keep the per-step ActorWorker -> RolloutWorker weight update to support Online model updates.

RLinf provides a high-performance, easy-to-use asynchronous communication abstraction, Channel, which adaptively uses optimized point-to-point backends (e.g., CUDA IPC and NCCL) and exposes a producer-consumer queue pattern. Thus ServerRolloutWorker -> InferenceWorker can be implemented as follows:

1. Create a channel. We reuse the naming convention in math. In Runner.__init__, call `self.dataloader_channel = Channel.create("DataLoader")`.
2. In Runner.run, call

.. code-block:: python

   rollout_handle: Handle = self.server_rollout.rollout(
       output_channel=self.dataloader_channel,
   )

   infer_handle: Handle = self.inference.run_inference(
       input_channel=self.dataloader_channel,
       output_channel=self.inference_channel,
       compute_ref_logprobs=self.compute_ref_logprobs,
   )

This implements ServerRolloutWorker -> InferenceWorker and greatly simplifies the code.

4. Algorithm and Offline Validation
---------------

Through reinforcement learning, we can convert the objective of user acceptance rate into a reward definition and internalize the policy that optimizes this reward into the model, obviating the need for a separate model to predict acceptance.

For the commonly used RL algorithms in LLM RLVR tasks—PPO and GRPO—each has practical difficulties here:
- PPO is based on Actor-Critic, which requires a stronger critic model than the policy model, increasing complexity and compute.
- GRPO avoids a critic by using group sampling to directly compute advantage. However, our scenario cannot afford multi-sampling and evaluation per request.

Cursor simplifies from the policy-based RL basics. Suppose the reward is \(J(θ)=E_{s∼P(s),\,a∼π(a∣s,θ)}[R(s,a)]\). If we assume \(R(s,a)\) is independent of \(θ\), the gradient is: \(\nabla_{θ}J(θ)=E_{s∼P(s),\,a∼π(a∣s,θ)}[\nabla_{θ}\,\log\,π(a∣s,θ)\cdot R(s,a)]\). We can compute \(\log\,π(a∣s,θ)\) and directly obtain \(R(s,a)\) from accepted vs. rejected, thus giving us an unbiased estimator of \(\nabla_{θ}J(θ)\). By tuning the step size, the impact of any dependence between \(R(s,a)\) and \(θ\) can be mitigated, enabling stable training.

As we lack a large-scale usage scenario, we demonstrate efficacy via offline RL. We build our own training and test sets and use an LLM-as-judge scoring scheme to simulate human preference.

code-fim-v2 is a code completion dataset across multiple languages. We select Python samples and filter out short completions, ending up with 4,000 high-quality samples. We use 3,000 for training and 1,000 for testing. Each sample provides the prefix and suffix of the code around the completion; the model generates the middle based on context.

For offline training, to simulate the Cursor online RL scoring approach, we do not use edit distance between completion and reference as the score. Instead, we use LLM-as-judge to rate each completion (0–10) to simulate human preference, and the average over all samples is the model’s score on the test set.

We use Qwen2.5-Coder-1.5B for experiments. We adopt a small learning rate (2e-6) and bf16 precision for stability: without KL loss, a larger LR can cause rapid forgetting early in training; bf16 stabilizes grad norms vs. fp16. For reward, we use LLM-as-judge with the same scoring prompt as evaluation; the judge model is deepseek-v3.1. For online RL we use PPO with group size=1 (as in AReaL, no critic); for offline training we use GRPO (group size=8) to quickly verify effectiveness on this task.

Scoring prompt for code completion results

.. code-block:: python

    You are a code quality evaluation expert. Please rate the given code completion result. The score will be used as the reward signal in RL training, so ensure objectivity, consistency, and discriminability.

    Information for evaluation
    <prefix>{prefix}</prefix>
    <suffix>{suffix}</suffix>
    <completion>{completion}</completion>

    Field descriptions
    prefix: The preceding part of the code
    suffix: The following part of the code
    completion: The LLM-provided completion content (the part between Prompt and Suffix).

    Scoring criteria (0–10), with five levels (0, 3, 6, 8, 10):
    Correctness and Functionality:
    0: Completely fails to achieve the intended functionality; fundamental logical errors
    3: Partially works but has severe logical flaws or fails common cases
    6: Achieves core functionality, with some minor mistakes or edge-case issues
    8: Correctly implements all functionality with only negligible issues
    10: Perfect functionality, rigorous logic, and robust handling of edge cases

    Please output results in the following XML format, ensuring the score is one of the specified five levels and the justification is concise and specific:
    ```xml
    <evaluation>
      <criteria_scores>
        <correctness_and_functionality>
          <score>[SCORE]</score>
          <justification>[Concise, specific reasoning]</justification>
        </correctness_and_functionality>
      </criteria_scores>
    </evaluation>
    ```

As shown below, the reward steadily increases during training, and the final test set score improves significantly (4.532 -> 6.897), a gain of over 50%, surpassing the 32B model in the same series. This indicates RL on the completion model is feasible, and small models show strong potential.

.. list-table::
   :widths: 50 50
   :header-rows: 0
   :align: center

   * - .. image:: https://github.com/RLinf/misc/raw/main/pic/coding_online_rl_offline_rewards.png
          :width: 100%
          :alt: Reward curve during training
     - .. list-table::
          :header-rows: 1
          :align: center

          * - Model
            - Score
          * - Qwen2.5-Coder-1.5B
            - 4.532
          * - Qwen2.5-Coder-3B
            - 5.139
          * - Qwen2.5-Coder-7B
            - 5.68
          * - Qwen2.5-Coder-14B
            - 6.351
          * - Qwen2.5-Coder-32B
            - 6.545
          * - Qwen2.5-Coder-1.5B-RL
            - 6.897 (+52%)
   * - Reward curve
     - Test set scores (0–10)

5. Conclusion
------

We built a working prototype loop of “online completion + reinforcement learning” on RLinf: without introducing costly critic or a separate reward model, we directly turn real user behavior into an optimization signal, and robustly coordinate “service and training” in a decoupled fashion. With low-friction data collection via VSCode/Continue, Channel pipelines, and online weight updates, this forms a minimal viable path for continuous iteration.

At a broader level, Online RL is likely to become the “continuous learning infrastructure” for human-in-the-loop AI products: converting observable user interaction into optimizable objectives and forming fast feedback–update loops in real environments. Cursor’s practice (about +28% acceptance improvement) and our offline validation jointly indicate that feedback-driven, online optimization is effective in code intelligence scenarios.

Following this trend, RLinf will continue to evolve into a general substrate for Online RL, while we extend this prototype to richer editor interactions and task forms, continuously optimize sampling and update strategies, and validate sustainable value and boundaries in real products.


