构建一个 Online RL 代码补全案例
============================

最后更新：10/20/2025。

相关阅读：:doc:`代码补全在线强化学习 <../examples/coding_online_rl>`。

1. 概览
------

在搜索与推荐等场景，Online Learning 已被验证有效。面向代码编辑器的智能补全，我们关心：能否把真实的用户“接受/拒绝”作为即时反馈，通过 Online RL 持续优化在线效果？我们参考 Cursor 团队的实践（见 `博文 <https://mp.weixin.qq.com/s/ShalRibfp9YSE5UFS0GLVg>`_），尝试在真实交互中提升补全质量与接受率。

为何该问题适合 Online RL：
1. 行为模式天然闭环：模型给出下一步建议，用户行为直接形成 reward 反馈；
2. 需求与场景持续变化，难以长期依赖“构建离线数据集 -> 低频离线训练”的路径；
3. 具备足量、可度量的实时用户反馈，可同时作为线上指标与训练信号。

基于上述判断，我们在 RLinf 框架上实现了一个可运行的原型系统（下文统一称“原型系统”，等同于常说的 Demo），聚焦“代码补全”任务：
- 集成 VSCode 侧的数据采集（基于 continue），在用户按下 Tab 或放弃时上报 prompt 与 accepted 标记；
- 在 RLinf 中打通在线推理/训练闭环，确保服务与训练可并行、互不阻塞；
- 针对补全任务做了一系列算法与工程上的轻量优化与尝试。

后续章节将依次介绍：任务与数据采集、系统实现（基于 RLinf）、算法与离线验证，最后给出结论与价值判断。需要说明的是，RLinf 在这里充当分布式执行与高性能通信的底座，使我们能在同一套框架内协同编排推理与训练。

2. 代码补全任务
-------------

对于如 Cursor 这样的编辑器，核心功能之一就是高效的代码补全能力。对于代码补全任务，最基础的功能为：当光标处在某个位置时，编程助手给出当前位置建议的插入内容。针对此功能，我们可以使用 FIM (Fill-In-the-Middle) 任务实现：给出上文和下文，大模型返回中间补全的建议。现如今大多数 llm 在预训练时已经针对 FIM 进行了训练，通过对模型输入 `<|fim_prefix|>上文<|fim_suffix|>下文<|fim_middle|>` 的 prompt，模型即可完成补全。

我们基于开源 ai 编程插件 continue 实现了该原型系统的数据采集。continue 默认已支持使用 FIM 任务格式提供代码补全能力，但并不支持人类反馈 reward 的上报。我们稍作改造：当用户按下 Tab 时，插件上报 prompt 与 accepted=True；当超过 10 秒未按下 Tab 或进行了其他代码操作，上报 prompt 与 accepted=False。由此我们直接获得人类反馈，可作为实时指标与训练信号，而无需额外训练一个 reward 模型。

对于 Cursor 实际使用场景，除了往光标处补全外，还支持了跨行补全、对另一段相似代码执行刚刚用户的修改操作等高级功能。很遗憾，这种能让人“一路火花带闪电狂按 tab”的高级能力不是 FIM 任务可以实现的，需要更高级的任务类型才能支持了。

3. 系统实现
----------

为承载“在线服务与训练并存”的目标，我们在 RLinf 上采用“推理与训练分离”的部署形态，并通过 `Worker`/`WorkerGroup` 进行组件抽象与编排。RLinf 框架提供了 Worker 的编程接口，这是其构建整个系统的基石。其中：

- Worker 表示一个远程进程或计算单元，通过继承 Worker 类，可以将一个具体的执行单元的逻辑进行抽象，并提供和其他 Worker 交互、及被 RLinf 分配和管理的能力。
- WorkerGroup 是一个用于创建和管理一组同类 Worker 的工具类，通过 Worker.create_group().launch(cluster, placement) 操作，可在集群资源上创建一组 Worker 实例，并且通过 WorkerGroup 接口可以进行远程调用。

math rl 训练代码的主要组件有：

- Runner：调度者，通过调用 RolloutWorker / InferenceWorker / ActorWorker 等不同组件，完成组件间配合以顺利执行代码
- RolloutWorker / InferenceWorker / ActorWorker：RL 训练中 rollout actor 的组件抽象。其中 inference 用于计算 pref_logprobs，以防止 actor 权重更新无法对后续数据计算 pref_logprobs

我们基于已有的 math rl 训练代码进行 online rl 的改造。框架层面的关键调整包括：

- 由于需要支持在线服务，使用共享式调度策略会导致训练时无法提供补全服务，因此必须使用分离式的调度策略。
- 保留 RolloutWorker / InferenceWorker / ActorWorker，但训练数据不再使用 RolloutWorker -> InferenceWorker -> ActorWorker 的流程，而是增加一个 ServerRolloutWorker 用于接受用户上报数据，训练数据的流程变为 ServerRolloutWorker -> InferenceWorker -> ActorWorker。
- 由于不再需要训练数据集，因此 RLRunner 中不再需要 training dataset 及各种处理，而是增加一个 OnlineRouterWorker 用于接受用户在线请求，使用 OnlineRouterWorker -> RolloutWorker 的流程提供线上服务。同时保留每步训练后 ActorWorker -> RolloutWorker 的权重更新过程，以支持 Online 的模型更新。

RLinf 框架提供了高性能、易用的异步通信抽象 Channel，自适应使用优化过的点对点后端（如 CUDA IPC 和 NCCL），并封装为生产者-消费者队列的通信模式。因此 ServerRolloutWorker -> InferenceWorker 可以如下实现：

1. 创建一个 channel。我们使用与 math 一致的命名，在 Runner.__init__中调用 `self.dataloader_channel = Channel.create("DataLoader")` 即可创建
2. 在 Runner.run 中调用

.. code-block:: python

   rollout_handle: Handle = self.server_rollout.rollout(
       output_channel=self.dataloader_channel,
   )

   infer_handle: Handle = self.inference.run_inference(
       input_channel=self.dataloader_channel,
       output_channel=self.inference_channel,
       compute_ref_logprobs=self.compute_ref_logprobs,
   )

即可实现 ServerRolloutWorker -> InferenceWorker 的逻辑，大大简化了代码逻辑使用。

4. 算法与离线验证
---------------

通过强化学习，我们可以使我们对用户接受率的目标转换为奖励设定，进而将针对奖励设定的策略通过强化学习训练进模型中，从而不再需要一个单独的模型来预测用户接受率。

对于当前大模型 RLVR 任务中最常见的 rl 算法 PPO 及 GRPO，使用起来分别有如下困难：
- PPO 基于 Actor-Critic 算法，因此需要一个相较于 policy 模型来讲更强的 critic 模型。这会导致 RL 过程复杂且计算量高。
- GRPO 通过使用 group 采样的方式，省去了 critic 模型直接计算 advantage。但我们的场景里无法做到多次采样且评估。

Cursor 从 policy-based RL 基础公式做了更简单的简化。假设奖励为 \(J(θ)=Es∼P(s), a∼π(a∣s,θ)[R(s,a)]\)，通过假设 \(R(s,a)\) 和 \(θ\) 无关，那么奖励的梯度为：\(\nablaθ J(θ)=Es∼P(s),a∼π(a∣s,θ)[\nablaθ log π(a∣s,θ)⋅R(s,a)]\)。其中 \(log π(a∣s,θ)\) 可以计算，\(R(s,a)\) 可以直接通过用户反馈的 accepted 与否，因此可以直接获得 \(\nablaθJ(θ)\) 的无偏估计。通过调整更新步长，\(R(s,a)\) 和 \(θ\) 有关对梯度的影响可以被缓解，此时模型可以正常训练。

由于我们没有足够大的使用场景，因此我们通过离线 rl 证明 rl 对补全任务的有效性。我们自己构建了训练集和测试集，使用大模型打分的方式来模拟人类偏好。

code-fim-v2 是一个包含多种编程语言的代码补全数据集，我们从中挑选出了 python 的补全样本，并进一步过滤掉补全内容较短的样本，最后剩下 4000 条高质量代码补全数据。取其中的 3000 条作为训练集，1000 条作为测试集。数据样本给出待补全代码片段的上文(prefix)和下文(suffix)，模型根据上下文的代码内容生成补全结果。

在离线训练中，为了模拟 cursor online rl 的打分方式，我们并未使用模型补全结果和参考答案的编辑距离作为分数，而是使用 llm as judge 对模型补全的结果进行评分（分数范围 0-10 分）来模拟人类偏好，所有样本的平均评分作为该模型在测试集上的分数。

我们采用 Qwen2.5-Coder-1.5B 模型进行实验。训练过程中，我们采用了较低的学习率 2e-6 和 bf16 数值精度以保证训练的稳定性：由于没有加 kl loss，较高的学习率可能导致模型训练初期遗忘过快；使用 bf16 训练相比 fp16 训练初期 grad norm 更稳定。reward 我们使用的是 llm as judge 的方式，打分 prompt 与测试集评测的打分 prompt 保持一致，打分模型使用 deepseek-v3.1。online rl 我们使用的是 group size=1 的 ppo(同 AReaL 的实现，无 critic model)，离线训练使用 GRPO(group size=8) 快速验证模型在该任务上的训练效果。

代码补全结果打分 prompt

.. code-block:: python

    请你作为代码质量评估专家，对给定的代码补全结果进行质量评分。这份评分将用于强化学习训练中的奖励信号，因此请确保评分客观、一致且有区分度。

    评估依据信息
    <prefix>{prefix}</prefix>
    <suffix>{suffix}</suffix>
    <completion>{completion}</completion>

    信息项描述
    prefix: 代码的前半部分
    suffix: 代码的后半部分
    completion: LLM 提供的待评估补全内容（即 Prompt 和 Suffix 之间的部分）。

    评分标准如下，采用 0-10 分制，分为 5 个等级（0, 3, 6, 8, 10）：
    正确性和功能性（correctness_and_functionality）：
    0 分：代码完全不能实现预期功能，存在根本性逻辑错误
    3 分：代码能实现部分功能，但存在严重逻辑缺陷或无法处理常见情况
    6 分：代码能实现核心功能，但存在一些边缘情况处理不当或 minor 错误
    8 分：代码能正确实现所有功能，仅存在极少可忽略的问题
    10 分：代码完美实现所有功能，逻辑严谨，能妥善处理各种边缘情况

    请基于以上标准对提供的代码补全结果进行评分，并按照以下 XML 格式输出，确保分数为指定的五个等级之一，理由简短具体且有针对性：
    ```xml
    <evaluation>
    <criteria_scores>
        <correctness_and_functionality>
        <score>[SCORE]</score>
        <justification>[简短具体的理由]</justification>
        </correctness_and_functionality>
    </criteria_scores>
    </evaluation>
    ```

如下图所示，模型在训练过程中reward稳步提升，训练结束在测试集上提升效果明显(4.532 -> 6.897)，涨幅超50%，超过同系列32B模型。由此可见对补全模型继续做rl是可行的，并且小模型也展现出了巨大的潜力。

.. list-table::
   :widths: 50 50
   :header-rows: 0
   :align: center

   * - .. image:: https://github.com/RLinf/misc/raw/main/pic/coding_online_rl_offline_rewards.png
          :width: 100%
          :alt: 训练reward变化图
     - .. list-table::
          :header-rows: 1
          :align: center

          * - 模型
            - 分数
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
   * - 训练reward变化图
     - 测试集得分（0-10 分）

5. 结语
------

我们在 RLinf 上跑通了“在线补全 + 强化学习”的原型闭环：无需引入昂贵的 critic 或额外的 reward 模型，直接将真实用户行为转化为优化信号，并以“服务与训练分离”的形态稳定联动；配合 VSCode/continue 的低摩擦数据采集、Channel 管道与权重在线更新，形成可持续迭代的最小可行路径。

更宏观地看，Online RL 有望成为人机协同类 AI 产品的“持续学习基础设施”：把可观测的用户交互转化为可优化目标，在真实环境中形成快速的反馈—更新闭环。Cursor 的实践（接受率约 +28% 提升）与我们的离线验证共同表明：反馈驱动、在线优化在代码智能场景切实有效。

顺应这一趋势，RLinf 将持续演进为面向 Online RL 的通用底座，同时我们将把该原型系统逐步拓展到更丰富的编辑器交互与任务形式，持续优化采样与更新策略，在真实产品中验证可持续的价值与边界。
