# YAM 双臂接入 RLinf 方案

本文记录 YAM 接入 RLinf 的原生实现边界。目标机器只需配置 RLinf 环境并安装固定版本的
`i2rt` SDK；运行时不下载、不导入，也不依赖 `yam-abc-reproduce`。本文不表示真机程序已经
通过验收；任何真机测试都应在清空工作区、急停可用且有人值守的条件下单独进行。

RLinf 机器人设备的通用组织方式、YAM 每个文件的职责和当前采集操作手册，见
[`yam_robot_device_organization.md`](yam_robot_device_organization.md)。

## 1. 结论

不要把旧采集脚本或另一个仓库塞进 RLinf worker。当前方案分为四层：

1. **RLinf 原生硬件描述**：`DualYamConfig` 描述 4 条 CAN 链和具名相机，scheduler 枚举
   只解析配置，不打开设备。
2. **RLinf 原生环境**：NUC 上唯一的 env worker 通过 `DualYamJointEnv` 独占 follower 与
   相机；固定使用 14D 双臂绝对关节动作。
3. **可选主臂接管**：只有采集/DAgger 模式才由 `DualYamLeaderIntervention` 打开 leader；
   纯策略模式不会打开主臂 CAN。
4. **数据和训练入口**：RLinf collector 已可直接消费该 Gym 环境并写 LeRobot；OpenPI
   dataconfig、SFT/eval 配置和真机验收仍是后续阶段。

目标调用链是：

```text
RLinf collector / rollout
  -> RealWorldEnv
  -> DualYamLeaderIntervention（采集/DAgger 时可选）
  -> DualYamJointEnv
  -> YamControlRuntime（唯一 follower command writer）
  -> lazy i2rt backend
  -> YAM follower/leader + cameras
```

数据闭环仍为：14D state/action + 3 路 RGB + task → LeRobot → norm stats → SFT →
离线 action sanity check → 纯策略真机评测 → DAgger（可选）。

## 2. 旧采集 pipeline（仅作迁移参考）

旧入口为：

```bash
python examples/embodiment/collect_yam_data.py \
  --station examples/embodiment/config/yam_station.yaml \
  --task pick_block \
  --episodes 50 \
  --convert
```

该入口依赖 `yam-abc-reproduce`，因此不再作为目标运行入口。迁移时仍应保留其中已经验证过的
以下行为：

1. 读取原生 `StationConfig`，不是 Hydra 配置。
2. 先启动并预热 3 个 RealSense worker，再打开 4 条 CAN 链。这个顺序必须保留，
   否则相机初始化阻塞可能触发电机通信 watchdog。
3. `ControlLoop` 以 30 Hz 读取两只 motorized leader，将关节目标镜像到左右 follower。
4. 顶部按钮切换同步；第二个按钮开始/停止一个 episode。
5. `EpisodeRecorder` 先写 canonical episode；`--convert` 在采集结束后批量转换为 LeRobot。

当前 canonical episode 已确认包含：

- 左右臂各 6 个关节状态和 1 个归一化夹爪状态；
- 左右臂各 6 个关节命令和 1 个归一化夹爪命令；
- `top`、`left`、`right` 三路 640×480 RGB 视频及各自时间戳；
- task、30 Hz、帧数、schema version 等 metadata。

旧 LeRobot converter 的输出契约为：

| 字段 | shape | 顺序/含义 |
|---|---:|---|
| `observation.state` | `(14,)` | `[L_q0..q5, L_grip, R_q0..q5, R_grip]` |
| `action` | `(14,)` | `[L_target_q0..q5, L_grip, R_target_q0..q5, R_grip]` |
| `observation.images.top_rgb` | video | 主视角 |
| `observation.images.left_rgb` | video | 左视角 |
| `observation.images.right_rgb` | video | 右视角 |
| `task` | string | 语言任务，如 `pick_block` |

RLinf 原生 collector 不再运行 `--convert`，而是直接写当前 RLinf LeRobot schema：

| 中间层/最终字段 | 含义 |
|---|---|
| raw `frames.top_rgb` | `RealWorldEnv.main_images` → LeRobot `image` |
| raw `frames.left_rgb` | 排序后的 `extra_view_images[0]` → `extra_view_image-0` |
| raw `frames.right_rgb` | 排序后的 `extra_view_images[1]` → `extra_view_image-1` |
| raw `state.joint_position` | `states` → LeRobot `state` |
| wrapper 实际接受动作 | `info["intervene_action"]` → LeRobot `actions` |

当前 writer 不在 LeRobot 字段名中保留 `left_rgb`/`right_rgb` 语义，训练 dataconfig 必须按
上表显式还原。一次采集同时产生 `logs/<timestamp>/demos/` 和
`logs/<timestamp>/collected_data/rank_0/id_0/`；`meta/info.json` 位于后一个 shard 内。

### 迁移后仍未闭环项

- 已增加 `requirements/install.sh embodied --env yam`，通过 YAM 环境清单统一安装固定 commit
  的官方 i2rt SDK，不会 clone YAM 应用仓库，也没有独立 wheel 路径。该固定版本仍需完成
  现场低速验收，并评估下述 SDK 修正。
- 官方 i2rt v1.3.3 的 `get_yam_robot()` 在 `MotorChainRobot` 构造失败时没有回收已经启动的
  CAN chain，且现场所需的 control-thread join 与多圈夹爪 limits 修正尚未进入该版本。
  真机验收前应固定包含这些修正的 i2rt build；RLinf adapter 在拿不到 robot handle 前无法
  从外层可靠回收 SDK 内部局部对象。
- 原生 writer、mock 环境和 schema 仍需做一条端到端采集测试；本次没有启动任何真机程序，
  因而尚未确认真机数据中的 `meta/info.json`、视频和动作时序。
- YAM exporter 与 RLinf 使用的 LeRobot 版本必须锁定并做兼容测试，尤其要避免 writer 写出
  新布局而 RLinf reader 只支持旧布局。
- 当前 schema 没有逐帧 CAN 延迟、相机同步误差、实际下发后的关节目标或 reject reason。
  做在线训练前必须补齐关键 telemetry。

## 3. 如何调“主臂重量/手感”

主臂手感由三个不同概念决定，不能只改一个 `weight`：

### 3.1 `bilateral_kp`：力反馈强度，不是重力补偿

RLinf 原生配置在每个 leader 的 `YamDeviceConfig` 中设置：

```yaml
left_leader:
  bilateral_kp: 0.0
right_leader:
  bilateral_kp: 0.0
```

- `0`：主臂 PD 增益清零，只做重力补偿，手感最自由；当前就是这个模式。
- `> 0`：使用 `YAM` 原生关节 `kp` 的比例值，把主臂拉向从臂实测姿态，产生双边力反馈。
  值越大，遇到从臂跟踪误差时阻力越明显。
- 它不会把下坠的主臂“托轻”。如果目标只是让主臂更轻，不应先增大它。

建议保持 `0` 完成第一版采集。如果确实需要力反馈，只在空场低速下从很小的值逐级试验，
每次只改一个量，并记录每个关节的峰值扭矩、跟踪误差和急停行为。

### 3.2 `ee_mass` 和 `gravity_comp_factor`：真正影响托举感

`ee_mass` 会替换末端组件质量，`gravity_comp_factor` 会逐关节缩放逆动力学重力矩。
YAM v1 的默认因子来自所固定 i2rt build 的 `robots/config/yam_v1.yml`：

```yaml
gravity_comp_factor: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]
```

判断方式：

- 松手后持续下坠：补偿偏小，优先检查 teaching handle 的实际质量/模型，再小幅提高主要承重
  关节的补偿因子。
- 松手后自己上抬：补偿偏大，应降低对应因子或修正末端质量。
- 某一姿态轻、另一姿态重：通常是质量、质心或模型不准，不能用全局比例掩盖。

RLinf 已将 `ee_mass` 和 `gravity_comp_factor` 放在每个 `YamDeviceConfig` 中；follower 夹爪
与 leader teaching handle 应分别标定，例如：

```yaml
left_follower:
  ee_mass: <measured-follower-mass>
left_leader:
  ee_mass: <measured-leader-handle-mass>
```

不要用同一个全局质量同时调主臂和从臂，否则会把两种不同末端的模型误差混在一起。

### 3.3 阻尼与摩擦

- `grav_comp_kd` 增大会让主臂更稳、更黏，但不会补足重力。已检查的 i2rt v1.3.3
  `yam_v1.yml` 默认值为 `[0.1, 0.1, 0.1, 0.3, 0.05, 0.05]`；配置为 `null` 时始终以所固定
  SDK build 和 `arm_type` 的模型配置为准，不应假设为全零。
- `coulomb_friction` 只有在 `use_coulomb_friction=True` 时生效；RLinf 配置默认关闭。
- 官方 i2rt v1.3.3 只公开 `use_coulomb_friction` 开关，没有公开逐设备数值覆盖
  `grav_comp_kd`/`coulomb_friction` 的构造参数。显式填写这两个向量时，RLinf 会拒绝使用
  不支持的 SDK，而不会修改 i2rt 私有数组。
- 如果主臂“能悬住但拖动发涩”，应检查阻尼、摩擦、线缆拖拽和关节机械摩擦，不要继续加
  重力补偿。

### 3.4 安全调参顺序

1. 保持 `bilateral_kp=0`，测量 leader teaching handle 的质量和质心。
2. 只连接一只 leader，在多个静态姿态观察下坠/上抬趋势及电机扭矩。
3. 先修正 leader 专属 `ee_mass`，再按关节小幅修正 `gravity_comp_factor`。
4. 确认能自然悬停后，再决定是否需要 `grav_comp_kd`。
5. 最后才增加 `bilateral_kp` 做力反馈，并验证 follower 阻塞、通信超时和急停。

不要在双臂、从臂和相机全部在线时首次试验新的重力或力反馈参数。

## 4. RLinf 原生接入架构

### 4.1 进程和硬件所有权

YAM NUC 上只允许一个 env worker 打开 CAN 与相机：

```text
GPU node                         YAM NUC
actor / rollout  <-- Ray -->     EnvWorker
                                 └─ RealWorldEnv
                                    └─ DualYamJointEnv
                                       ├─ YamControlRuntime
                                       ├─ lazy i2rt backend
                                       ├─ optional leaders
                                       └─ RLinf camera backends
```

- 训练节点不直接 import 或打开 i2rt/CAN。
- env worker 启动顺序仍是 camera warm-up → CAN → control loop。
- 纯策略评测只打开两只 follower；采集/接管模式才打开两只 leader。
- close/offload 必须释放控制循环、CAN、相机线程和 socket，并可重复调用。

### 4.2 第一版动作契约：14D joint absolute

第一版应与已有数据完全一致，采用 14D 绝对关节动作，不要同时切换到 TCP：

```text
[L_q(6), L_gripper(1), R_q(6), R_gripper(1)]
```

约束：

- arm joint 单位为 rad；gripper 固定为 `0=closed, 1=open`。
- action 是通过限位与 slew-rate 检查后实际接受的目标。
- observation 使用实测关节位置，不使用上一条 command 冒充状态。
- 当前 collector 保存 accepted expert action 和 actual state；policy action、控制周期、CAN 延迟和
  reject reason 尚未作为独立 LeRobot 字段持久化，这是在线训练前需要补的 telemetry。
- `action_space` 必须包含真实关节上下限；gripper 为 `[0, 1]`。

等 14D 的采集、SFT、评测全部闭环后，再另建版本化的 20D TCP rot6d 数据契约。不要把 14D
数据静默转换成 20D，也不要在一个 dataset repo 内混用两种 representation。

### 4.3 已落地的第一批文件

硬件调度：

```text
rlinf/robotics/robots/dual_yam.py
rlinf/robotics/robots/__init__.py
rlinf/robotics/__init__.py
```

`DualYamConfig` 包含左右 follower/leader CAN channel、arm/gripper type、各设备 `ee_mass`、
gripper limits、相机 serial 和 `node_rank`。枚举过程不导入 i2rt、不探测 CAN，也不会写死
NUC IP 或训练机路径。

环境：

```text
rlinf/envs/real/yam/types.py
rlinf/envs/real/yam/config.py
rlinf/envs/real/yam/control_runtime.py
rlinf/envs/real/yam/i2rt_backend.py
rlinf/envs/real/yam/mock_backend.py
rlinf/envs/real/yam/dual_yam_joint_env.py
rlinf/envs/real/yam/leader_intervention.py
rlinf/envs/real/yam/tasks/__init__.py
rlinf/envs/real/yam/__init__.py
```

以 `DOSW1Env` 的 leader/follower 状态机和 `DualFrankaEnv` 的双臂观测契约为参考，注册
`DualYamJointEnv-v1`，再由现有 `RealWorldEnv` 完成 vector-env 包装和统一 observation：

```text
raw state dict + camera dict
  -> RealWorldEnv._wrap_obs
  -> states / main_images / extra_view_images / task_descriptions
```

已增加的采集配置：

```text
examples/embodiment/config/env/realworld_dual_yam_joint.yaml
examples/embodiment/config/realworld_dual_yam_collect_data.yaml
```

启动复用未修改的 RLinf 既有入口：

```bash
bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data
```

安装入口 `requirements/install.sh embodied --env yam` 会安装 RLinf embodied、RealSense、
OpenCV、固定 LeRobot，以及 `requirements/embodied/envs/yam.txt` 中固定 commit 的官方
`i2rt` SDK。入口不会 clone 浮动 `main`，也不依赖 NUC 上的嵌套 YAM 应用仓库，没有额外
wheel 变量或独立 SDK 安装路径。代码仍只在真实 backend 的 `connect()` 时延迟导入
`i2rt`，因此 scheduler 和 dummy 环境的模块导入保持无硬件副作用。OpenPI 策略评测配置
尚未增加，因为当前代码还没有 YAM 专用 14D dataconfig，不能伪装成已闭环。

## 5. 数据与 OpenPI 接入

### 5.1 先验收 exporter

在接 RLinf loader 前，至少用 3 个短 episode 验证：

1. `collected_data/rank_0/id_N/meta/info.json` 存在，LeRobot 能列出 episode 和 task；
2. state/action shape 恒为 14，无 NaN/Inf；
3. 三路视频帧数、时间戳和控制 step 对齐，报告最大与 p95 偏差；
4. action 的左右臂顺序、关节符号和夹爪开合方向与真机一致；
5. episode 末尾没有重复旧相机帧或缺失 action；
6. 用 action 回放到 mock/sim 时轨迹连续且不越限。

### 5.2 新建 YAM dataconfig

不要直接使用当前 `pi0_realworld`：它的 transform 明确假设 19D state 和 7D action，和 YAM
14D/14D 不兼容。应新增 YAM 专用配置，例如：

```text
rlinf/models/embodiment/openpi/dataconfig/yam_dataconfig.py
rlinf/models/embodiment/openpi/policies/yam_policy.py
```

映射建议固定为：

```text
top_rgb   -> base_0_rgb
left_rgb  -> left_wrist_0_rgb
right_rgb -> right_wrist_0_rgb
state     -> 14D
actions   -> [horizon, 14]
```

随后在 OpenPI config registry 注册 `pi0_yam`/`pi05_yam`，增加一份 SFT YAML，并确保：

- `actor.model.action_dim: 14`；
- `num_action_chunks` 与 OpenPI `action_horizon` 完全一致；
- norm stats 从当前 YAM dataset 重新计算，不能复用 Franka/Aloha stats；
- task string 与部署时 prompt 完全一致；
- dataset metadata 写入 action representation 和 schema version。

## 6. 从 SFT 到在线 DAgger

按以下门禁推进，前一项未通过时不要进入下一项：

### M0：冻结依赖与数据契约

- 将 RLinf 内的 YAM 改动形成可追踪提交；
- 固定 i2rt、LeRobot commit/version；
- 冻结 14D joint absolute 顺序、单位、夹爪方向和 3 个 camera key；
- 将 leader/follower 的 `ee_mass` 分开配置。

### M1：离线数据闭环

- mock 采集和转换测试；
- 真机 3 条短 episode 转换；
- RLinf loader smoke test；
- norm stats；
- 小数据 overfit，确认模型输出维度、范围和连续性。

### M2：原生 Env 的 dummy/sim 闭环

- `DualYamConfig`/`DualYamRobot` 注册和 config 解析测试；
- `DualYamJointEnv-v1` reset/step/close 测试；
- joint limits、slew limit、NaN、timeout、重复 close 测试；
- 相机缺帧和 CAN 异常必须返回结构化错误并进入 hold/estop。

### M3：真机纯策略评测

- 单臂、低速、无物体评测；
- 两臂分区评测；
- policy server/rollout 超时后保持当前目标，绝不发送零位；
- 验证 action chunk 只按受控频率消费，禁止一次性把整个 chunk 瞬间下发；
- 通过后才进行任务评测。

### M4：leader intervention / DAgger

- 复用或泛化 `LeaderFollowerKeyboardIntervention`；
- 明确 MODEL/PAUSE/TELEOP/ESTOP 状态机；
- 接管时先将 follower 平滑对齐 leader，禁止跳变；
- `info["intervene_action"]` 写入实际接受的完整 14D action；
- 额外保留 left/right intervention mask，单手接管不能错误覆盖另一只手；
- 先离线检查 replay buffer，再允许 actor 更新。

## 7. 最低测试矩阵

| 层级 | 测试 | 通过标准 |
|---|---|---|
| config | YAM hardware YAML 解析 | channel、serial、质量、limits 无静默默认错误 |
| robot adapter | state/action round-trip | 14D 顺序、单位、夹爪极性完全一致 |
| safety | NaN、越限、大步长、超时 | reject/hold，不下发危险 command |
| lifecycle | init/reset/close×2 | 无残留 CAN、相机或控制线程 |
| dataset | 3 episode convert/load | shape、task、视频、时间戳完整 |
| SFT | 小数据 overfit | loss 下降，输出范围合理，无维度 padding 错位 |
| dummy env | RLinf collector/eval | 无硬件也能跑通接口和 episode lifecycle |
| hardware | 单臂低速 | 无跳变、无 watchdog、急停有效 |
| dual arm | 分区动作 | 左右 action slice 不串扰、不碰撞 |
| DAgger | 单手/双手接管 | mask、expert action、policy action 语义正确 |

## 8. 实现状态与后续拆分建议

当前工作区已经完成硬件配置、原生 joint env、i2rt 边界、dummy backend、主臂接管、
RLinf 原生采集 YAML、YAM 安装入口和对应单元测试；未运行任何真机程序。后续建议按可独立
审查的提交拆分：

1. `docs: document YAM integration contract`
2. `feat(realworld): add RLinf-native DualYam environment`
3. `build: add pinned i2rt YAM environment dependencies`（当前工作区已纳入 `--env yam`）
4. `feat(realworld): add YAM collector and station configs`（当前工作区已实现）
5. `feat(data): add YAM LeRobot dataconfig`
6. `test(data): validate YAM 14D dataset contract`
7. `feat(embodiment): add YAM SFT and eval configs`

依赖安装、dataset schema、真机验收和在线 DAgger 应保持可独立回退。出现问题时才能明确
区分数据、模型、调度、CAN 或遥操状态机。
