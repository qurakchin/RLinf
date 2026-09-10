# RLinf 机器人设备组织与 Dual YAM 接入说明

本文说明 RLinf 如何声明、枚举、调度和控制真实机器人，并以当前 Dual YAM
实现为例解释各层职责、采集入口、数据契约和后续接入 OpenPI 的工作。本文描述的是
RLinf 当前工作区中的原生实现；它不会调用或依赖另一个 YAM 应用仓库。

## 1. 总体结构：两条链和三类配置域

RLinf 将机器人接入拆成两条链：scheduler 资源链回答“哪台 Worker 可以使用哪套设备”，
runtime 控制链回答“这套设备如何连接、收发动作和生成观测”。

```text
scheduler 资源链
Hydra YAML
  -> ClusterConfig / NodeGroupConfig
  -> NodeHardwareConfig
  -> 具体机器人 HardwareConfig
  -> Hardware.enumerate()
  -> HardwareResource / HardwareInfo
  -> ComponentPlacement
  -> WorkerInfo.hardware_infos

runtime 控制链
EnvWorker 或 DataCollector
  -> RealWorldEnv
  -> Gym task factory
  -> 具体机器人 Gym Env
  -> 可选遥操/接管 Wrapper
  -> Controller 或 ControlRuntime
  -> SDK backend
```

这两条链对应三个配置域。

### 1.1 物理设备配置域

位置：`cluster.node_groups[].hardware`。

它描述相对稳定的设备身份和物理标定，例如：

- 设备属于哪个 `node_rank`；
- 机器人 IP、CAN channel、相机 serial；
- 夹爪类型和物理行程；
- 末端质量、重力补偿、摩擦补偿等设备标定。

YAM 中对应 `DualYamConfig`、`YamDeviceConfig` 和 `YamCameraConfig`。

### 1.2 任务与运行时配置域

位置：`env.train.override_cfg` 或 `env.eval.override_cfg`。

它描述一次任务如何运行，例如：

- 任务文本；
- 控制频率和最大 episode 长度；
- RLinf 侧关节安全限位和单步 slew limit；
- feedback、相机 warm-up、单帧和 stale timeout；
- 是否启用主臂、按钮防抖和非同步时采用 policy 还是 hold。

YAM 中对应 `DualYamJointEnvConfig` 和 `YamLeaderInterventionConfig`。

### 1.3 调度与采集配置域

位置：`cluster.component_placement`、`runner` 和 `env.*.data_collection`。

它描述：

- env Worker 使用哪个 node group 的第几个硬件资源；
- 采集多少个 episode；
- 数据写到哪里、采用什么格式和帧率；
- 是否只保留成功 episode，以及 LeRobot writer 的 finalize 策略。

三类配置应保持边界清晰。CAN 名称不应藏在任务配置中，任务文本也不应放进 scheduler
硬件枚举代码中。

## 2. scheduler 资源链

### 2.1 基础类型

基础定义位于 `rlinf/scheduler/hardware/hardware.py`。

#### `HardwareConfig`

所有设备配置 dataclass 的父类，至少包含：

```python
node_rank: int
```

它表示设备由哪台 Ray 节点访问，而不是 Worker rank 或硬件编号。

#### `NodeHardwareConfig`

对应 YAML 中一个 node group 的 `hardware` 字段：

```yaml
hardware:
  type: DualYam
  configs:
    - node_rank: 0
      ...
```

机器人类型通过 `Robot.register_type(config_cls, discovery_cls)` 注册，它在内部调用
`NodeHardwareConfig.register_hardware_config()` 建立 `type` 到具体配置 dataclass 的映射，
并把 robot class、config class、discovery class 和 builder 串成一条注册记录。
YAML 中的字典随后会被转换成 `DualYamConfig`、`DualFrankaConfig` 等强类型对象，并检查：

- 配置字段是否合法；
- 是否存在完全重复的配置；
- 每个 `node_rank` 是否位于所属 node group 中；
- 同一 `(node_rank, hardware type)` 是否被不同 node group 重复声明。

虽然基础类 docstring 提到类型不区分大小写，但当前注册表实际使用精确字符串查找，因此 YAM
配置必须写成 `type: DualYam`。

#### `Hardware`

每种设备的枚举策略父类。具体机器人通过 `@Hardware.register()` 注册，并实现：

```python
enumerate(node_rank, configs) -> HardwareResource | None
```

枚举的含义是把配置转换成可调度资源；它不等同于连接和启动机器人。不同实现可以做只读
可见性检查，但不应在 Cluster 探测阶段发送运动命令。

#### `HardwareInfo` / `RobotInfo`

一个可调度硬件实例的描述，基础字段为 `type` 和 `model`。机器人使用泛型化的
`RobotInfo[RobotConfigType]`，其中的 `config` 就是注册时绑定的配置类型，例如：

```python
RobotInfo.type: "DualYam"
RobotInfo.config: DualYamConfig
```

`HardwareInfo` 是 scheduler 最终交给 Worker 的对象；`RobotInfo` 是机器人这一层的具体形态，
环境通过 `robot_info.config` 取到强类型配置。

#### `HardwareResource`

同一节点上一组相同类型的 `HardwareInfo`：

```python
HardwareResource(type="DualYam", infos=[...])
```

`count` 等于 `len(infos)`。一条 `HardwareInfo` 是一个调度和独占单元，但不一定是一条机械臂：
Dual YAM 的一个 info 表示一整套双 follower、双 leader 和相机工作站。

### 2.2 NodeProbe 如何形成资源视图

关键文件：

- `rlinf/scheduler/cluster/config.py`
- `rlinf/scheduler/cluster/node.py`
- `rlinf/scheduler/cluster/cluster.py`

Cluster 初始化时会在每台 Ray 节点上启动 `_RemoteNodeProbe`：

1. 读取该节点的 `RLINF_NODE_RANK`；单节点时强制为 0。
2. 从 `ClusterConfig` 取出所有属于该节点的 `HardwareConfig`。
3. 遍历 `Hardware.policy_registry`，逐一调用已注册策略的 `enumerate()`。
4. 将非空结果放入 `NodeInfo.hardware_resources`。
5. 根据 YAML 中的 node group 构造 `NodeGroupInfo`。

显式机器人 node group 的 `hardware_type` 就是 YAML 的 `hardware.type`。未显式声明硬件的
node group 通常优先使用自动探测的 accelerator；没有 accelerator 时才按 node 进行调度。

### 2.3 node rank、hardware rank 和 local hardware rank

三种 rank 含义不同：

- `node_rank`：Cluster 中的机器编号。
- hardware rank：所选 node group 中的设备资源编号。
- local hardware rank：同一节点内部的设备资源下标。

例如节点 0 上配置两个完整 YAM station，节点 1 上配置一个：

```text
node_rank 0: local hardware rank 0, 1
node_rank 1: local hardware rank 0

node group hardware rank: 0, 1, 2
```

同一节点内 `HardwareInfo` 的次序来自配置和枚举结果的次序。因此
`component_placement.env.placement: 0` 表示 node group 中第一个完整 station，不表示
`node_rank=0`，也不表示 `can0`。

### 2.4 Placement 如何把设备交给 Worker

关键文件：

- `rlinf/scheduler/placement/placement.py`
- `rlinf/scheduler/placement/flexible.py`
- `rlinf/scheduler/worker/worker_group.py`
- `rlinf/scheduler/worker/worker.py`

配置：

```yaml
component_placement:
  env:
    node_group: yam
    placement: 0
```

会经过以下步骤：

1. `ComponentPlacement` 将硬件 rank 表达式解析为进程到资源的映射。
2. `MultiNodeGroupResolver` 找到资源所在 node、node group 和 local hardware rank。
3. `FlexiblePlacementStrategy` 生成 `Placement`。
4. `WorkerGroup` 启动 Ray Actor 时注入：

   ```text
   CLUSTER_NODE_RANK
   LOCAL_HARDWARE_RANKS
   NODE_GROUP_LABEL
   ```

5. `Worker._setup_hardware()` 恢复这些信息。
6. `Worker.hardware_infos` 从对应 `NodeGroupInfo` 中按 local rank 取出已分配设备。
7. `_setup_worker_info()` 将它们放入 `WorkerInfo.hardware_infos`。

具体机器人环境因此不需要重新读取 cluster YAML，也不应自行扫描并挑选“第一台机器人”。

## 3. runtime 控制链

### 3.1 `EnvWorker` / `DataCollector` 到 `RealWorldEnv`

训练时，`rlinf/workers/env/env_worker.py` 中的 `EnvWorker.init_worker()` 调用：

```text
get_env_cls("realworld") -> RealWorldEnv
```

当前采集入口复用 `examples/embodiment/collect_real_data.py`：

```text
DataCollector Worker -> RealWorldEnv
```

`RealWorldEnv` 当前要求每个 Worker 只创建一个真实环境。在创建内部 Gym 环境时，它执行
等价于：

```python
robot_info = worker_info.hardware_infos[env_idx]
gym.make(
    id=cfg.init_params.id,
    override_cfg=override_cfg,
    worker_info=worker_info,
    robot_info=robot_info,
    env_idx=env_idx,
    env_cfg=cfg,
)
```

因此：

- `env_type: realworld` 只选择通用的 `RealWorldEnv`；
- `init_params.id: DualYamJointEnv-v1` 才选择 YAM Gym task；
- 非 dummy YAM 必须通过硬件 node group 的 hardware placement 获取 `RobotInfo`；
- 使用普通 node placement 会得到空的 `hardware_infos`，真机 YAM 环境会拒绝启动。

### 3.2 Gym task factory 的装配职责

`rlinf/envs/real/yam/tasks/__init__.py` 注册 `DualYamJointEnv-v1`。factory 负责：

1. 从 `override_cfg` 分离基础 env 配置和 leader intervention 配置；
2. 创建 `DualYamJointEnv`；
3. 检查 `main_image_key` 是否是已配置相机；
4. 仅在 `leader_intervention.enabled=true` 时套上 `DualYamLeaderIntervention`。

这使纯策略评测不会连接两只 leader。

### 3.3 动作和观测的往返链路

```text
policy/collector placeholder action
  -> RealWorldEnv.step
  -> DualYamLeaderIntervention（可选：policy / hold / leader）
  -> DualYamJointEnv.step
  -> YamControlRuntime.command
  -> i2rt follower backend

i2rt measured state + cameras
  -> YamControlRuntime.read_state
  -> DualYamJointEnv observation
  -> RealWorldEnv._wrap_obs
  -> states / main_images / extra_view_images / task_descriptions
```

人工接管时 wrapper 返回 `info["intervene_action"]`。它是经过 RLinf 关节限位和 slew limit
后实际接受的完整动作；`RealWorldEnv` 会将其转成 tensor，并由采集器保存为 expert action。

## 4. 现有机器人实现对比

| 机器人 | scheduler 资源粒度 | 控制进程/位置 | 枚举特点 | 与 Dual YAM 的差异 |
|---|---|---|---|---|
| DOSW1 | 一套双臂主从工作站 | env 内直接创建 gRPC SDK adapter | 配置驱动，可用 `RobotAutoConfig` 从环境变量补字段 | 资源粒度最接近 YAM，但设备连接、回零和较多主从状态直接位于 env 中 |
| DualFranka | 一套双 Franka 工作站 | env 启动两个 Controller Worker；左右控制器可放在不同节点 | 配置驱动，可从环境变量补 IP | 支持相机 env 节点与机械臂控制节点分离；YAM 当前要求四条 CAN 和相机都可由 env 节点访问 |
| GimArm | 一只机械臂 | env 启动 `GimArmController` Worker，可指定 controller node | 配置驱动；枚举时只读检查 CAN interface 是否存在 | YAM 把四个 CAN 设备组合成一个 station，并在 env 进程内保持单一 follower 写者 |
| Turtle2/XSquare | 一套较粗粒度 Turtle2 资源 | env 启动同节点 ROS smooth controller | scheduler config 基本只有 `node_rank` | arm/camera 选择主要仍由 env 配置承担；YAM 在 scheduler 中完整声明设备拓扑和相机身份 |
| DualYam | 双 follower + 双 leader + 相机的完整 station | env 进程中的一个 `YamControlRuntime` | 完全配置驱动；不探测或打开 CAN，不导入 i2rt；检查跨 station 资源复用 | 以单写者、安全启动、按需连接 leader 和无应用仓库依赖为主要设计目标 |

当前 YAM 没有仿照 Franka/GimArm 再启动远程 Controller Worker。这样可以让一个 runtime
统一串行化两个 follower 的全部命令，避免多个 writer 竞争 CAN；代价是 station 当前不能跨节点。

## 5. 本次 YAM 文件逐项说明

### 5.1 robotics 机器人文件

#### `rlinf/robotics/robots/dual_yam.py`

YAM 物理工作站的声明和资源注册：

- `YamDeviceConfig`：单个 follower/leader 的 channel、arm/gripper 类型、末端质量、物理夹爪
  行程、重力/阻尼/摩擦参数、leader 双边反馈和夹爪方向。
- `YamCameraConfig`：稳定 camera key、serial、类型、分辨率、FPS 和 depth 开关。
- `DualYamConfig`：`RobotConfig` 子类，将四个设备与相机组合成一个完整 station，并递归转换
  Hydra mapping。因为基类自带 `disable_validate` 默认值，每个字段都必须有默认值，缺失由
  `__post_init__` 用 `ValueError` 拒绝。
- `DualYamRobot`：`Robot` 子类，`ROBOT_TYPE = "DualYam"`，不携带 builder。
- `DualYamDiscovery`：`RobotDiscovery` 子类，先交给基类枚举，再拒绝同节点 station 间的
  CAN 或相机 serial 复用。
- 模块底部调用 `DualYamRobot.register_type(DualYamConfig, DualYamDiscovery)`，把四者串成
  一条注册记录。

枚举明确是 configuration-only，不会导入 i2rt 或打开 SocketCAN。

`gripper_limits` 是有方向的物理标定，顺序固定为 `[closed, open]`，不能排序。由于电机
安装方向不同，它既可以递增，也可以递减，例如 `[1.2, -0.4]` 合法；只有两个端点相等才
非法。无论物理方向如何，RLinf 上层的归一化夹爪语义始终是 `0=closed, 1=open`。

#### robotics exports

- `rlinf/robotics/robots/__init__.py`：导入 YAM 模块，触发 `register_type`，并导出配置类型。
- `rlinf/robotics/__init__.py`：把 YAM 类型加入 robotics 的公共 import 路径。
- `rlinf/envs/real/__init__.py` 的 `_ROBOT_PACKAGES`：让指定 `DualYam` 的集群配置在
  不导入 i2rt 的前提下也能解析。

这些文件不控制机器人，只负责注册和公共 import 路径。

### 5.2 `rlinf/envs/real/yam` 文件

#### `types.py`

定义稳定的状态和 backend 契约：

- `YamArmState`、`YamLeaderState`、`DualYamState`；
- `YamCommandResult`；
- 14D pack/split 工具；
- `YamFollowerBackend`、`YamLeaderBackend`、`YamBackendFactory` Protocol。

它把上层 runtime 与 i2rt 具体对象解耦，也是 mock backend 能替代真机 backend 的基础。

#### `config.py`

定义任务级 `DualYamJointEnvConfig` 和 `YamLeaderInterventionConfig`，并验证控制频率、joint
limits、slew、超时、图像尺寸、按钮防抖和 `hold/policy` 取值。

#### `i2rt_backend.py`

唯一允许接触 i2rt 的边界：

- 模块顶层不 import i2rt；
- `_build_yam()` 只在 backend `connect()` 时 import `get_yam_robot`、`ArmType` 和
  `GripperType`；
- 将 scheduler 设备配置转成 i2rt 构造参数；
- follower 读写标准化 7D 状态、读取 SDK joint limits 并提供 measured-pose hold；
- leader 读取 teaching handle、学习按钮 idle 极性、处理夹爪方向并提供可选双边反馈；
- 集中封装当前 i2rt 缺少公开 health snapshot 时的兼容访问。

训练节点、scheduler 解析和 dummy 测试因此都不需要安装 i2rt。

#### `mock_backend.py`

内存 follower/leader 实现，不访问 CAN。用于 `is_dummy`、runtime/env 单元测试和按钮状态注入。

#### `control_runtime.py`

四个 transport 的唯一所有者和两个 follower 的单一写者：

- follower 和 leader 分阶段连接；
- 每只 follower 连接后立即 hold，再连接下一只；
- 启动时核对配置 joint limits 位于 SDK limits 内；
- 校验 14D shape、finite、hard limits、`max_joint_delta` 和 `[0,1]` gripper；
- NaN、stale feedback、读写异常和相机后处理异常走 measured-pose hold；
- `engage()` 将 follower 平滑对齐到当前 leader，避免接管跳变；
- 同步结束或异常时释放 leader 双边反馈；
- 连接或 close 部分失败时保留句柄，允许清理重试。

#### `dual_yam_joint_env.py`

面向 Gym/`RealWorldEnv` 的基础双 follower 环境：

- 固定 14D absolute-joint action/observation space；
- 解析 scheduler 下发的 station config，并验证 env Worker 与硬件 node 一致；
- 构造无硬件副作用；第一次 `reset()` 才执行相机 open/warm-up、follower connect、hold；
- 从 runtime 获取实测状态，而不是用上一条 command 冒充 observation；
- 将相机 BGR 转 RGB并调整到配置尺寸；
- 短暂丢帧可复用上一帧，超过 stale timeout 则抛错并 hold；
- 为 wrapper 提供 `teleop_tick()`、`observe()` 和 `get_hold_action()`。

#### `leader_intervention.py`

主臂遥操与 episode 控制 Gym Wrapper：

- 仅启用 wrapper 时连接 leader；
- 任一主臂的 top button 可切换 follower 同步；
- 任一主臂的 record button 可开始或成功结束 episode；
- 按钮按 rising edge 和 debounce 处理；
- 同步时用 leader action 替换 policy action；
- 非同步时可选择 `hold` 或 `policy`；
- 用 `intervene_action` 返回实际接受的 expert action；
- 同步关闭、episode 结束、reset 和异常时 hold follower 并释放 leader feedback。

#### `tasks/__init__.py`

注册 `DualYamJointEnv-v1` 并装配基础 env、camera key 校验和可选 leader wrapper。

#### `yam/__init__.py`

YAM env 包的公共导出；导入 `tasks` 以确保 Gym registration 在 `gym.make()` 前完成。

#### `rlinf/envs/real/__init__.py`

把 YAM 类型和 task registration 接入整个 RealWorld 包。该文件也导出现有 Franka、DOSW1、
GimArm 和 Turtle2 类型。

### 5.3 examples 文件

#### `examples/embodiment/config/env/realworld_dual_yam_joint.yaml`

可复用的 YAM env 模板：

- `env_type: realworld`；
- `init_params.id: DualYamJointEnv-v1`；
- `main_image_key: top_rgb`；
- YAM v1 nominal joint limits、slew、feedback/camera timeout 和输出尺寸；
- 默认关闭 leader intervention，非同步动作来源为 `policy`，适合后续纯策略评测。

#### `examples/embodiment/config/realworld_dual_yam_collect_data.yaml`

单 station、三相机、50 episode 的完整采集配置：

- `hardware.type: DualYam`；
- 四个设备和三台相机；
- follower gripper 的 `[closed, open]` 实测电机弧度由环境变量提供；
- leader intervention 开启，非同步时采用 measured-pose `hold`；
- `manual_episode_control_only: true`；
- 采集 `max_episode_steps: 10000`、`max_steps_per_rollout_epoch: 10000`；
- 直接写 LeRobot、`fps: 30`、`only_success: true`；
- `finalize_interval: 0`，干净退出时一次 finalize，保持单个 shard；
- `resume: false`。

`manual_episode_control_only` 使基础 env 和 `RealWorldEnv` 不用步数自动结束当前采集，正常结束
由 record button 负责。`max_episode_steps=10000` 仍作为 trajectory builder 的有限容量上限。

#### `examples/embodiment/collect_data.sh`

YAM 直接复用仓库现有的通用采集启动器，不修改该脚本。它的既有行为是：

- 第一个位置参数选择 Hydra config；未提供时使用 `realworld_collect_data`；
- 自动设置 `EMBODIED_PATH`、`REPO_PATH`、`SRC_FILE`、`PYTHONPATH` 和
  `HYDRA_FULL_ERROR`；
- 为每次运行建立带时间戳的 log 目录并保存完整命令和 stdout/stderr。

它仍调用已有 `collect_real_data.py`，没有新增 YAM 专用 Python 入口，也不会调用另一个 YAM
应用仓库的 collector/converter。

#### `requirements/install.sh`

新增 `embodied --env yam` 环境目标，用于统一安装 RLinf embodied、RealSense/OpenCV、固定
LeRobot，以及 `requirements/embodied/envs/yam.txt` 中固定 commit 的官方 i2rt SDK。
安装器不会 clone YAM 应用仓库，也没有额外的 wheel 环境变量或第二套安装入口。

#### `requirements/embodied/envs/yam.txt`

YAM 真机环境的专属依赖清单。它把 i2rt 固定到已经对接的官方 commit，并包含原生 runtime
所需的 OpenCV、RealSense 和 YAML 依赖。i2rt 的 `ruckig` 构建约束放在同目录的
`yam-build-constraints.txt`，只由 `install_yam_env` 使用，不修改 RLinf 根 `pyproject.toml`。

### 5.4 测试文件

- `tests/unit_tests/test_yam_hardware.py`：配置转换、station 枚举、重复资源、降序
  `[closed, open]` 和相同端点拒绝。
- `tests/unit_tests/test_yam_runtime.py`：连接、14D action、安全限制、hold、异常和清理。
- `tests/unit_tests/test_yam_env.py`：Gym env、dummy/backend 注入、动作和观测契约。
- `tests/unit_tests/test_yam_intervention.py`：按钮、同步、接管、policy/hold 和 episode 语义。
- `tests/unit_tests/test_yam_imports.py`：确认 scheduler/普通 import 不会提前 import i2rt。
- `tests/unit_tests/test_yam_examples.py`：静态检查两个 YAML 的 canonical contract、完整 station、
  `max_episode_steps=10000`、`finalize_interval=0` 和无外部应用仓库依赖。

这些测试使用 mock/stub，不构成真机验收。

### 5.5 文档文件

- `docs/yam_integration_plan.md`：接入范围、主臂手感参数、14D/OpenPI 迁移门禁和后续里程碑。
- `docs/yam_robot_device_organization.md`：本文，集中说明 RLinf 设备组织、文件职责和当前采集
  操作契约。
- `docs/source-en/rst_source/examples/embodied/yam.rst` 与 `docs/source-zh/.../yam.rst`：面向
  用户的英文/中文采集手册。
- 两侧 `examples/real_world_index.rst`：把 YAM 手册加入真实机器人文档目录。
- `README.md` 与 `README.zh-CN.md`：在功能列表中链接原生 YAM 数据采集入口。

## 6. 当前采集操作

### 6.1 软件和设备前提

当前 Python 环境通过 YAM 安装目标获得 RLinf 运行依赖、LeRobot、RealSense backend，以及
固定的 RLinf-compatible i2rt SDK。无需另行安装 i2rt，也不需要 clone 或设置任何
`yam-abc-reproduce`、YAM collector 或 YAM converter 仓库路径。

启动前还应确认四个 SocketCAN interface 已由系统配置并处于可用状态，三台相机 serial
正确。scheduler 枚举不会打开 CAN 或验证设备在线；真正的连接错误会在第一次 `reset()`
暴露。

安装命令为：

```bash
bash requirements/install.sh embodied --env yam
source .venv/bin/activate
```

该命令会在同一环境中安装固定的 i2rt SDK；没有额外的 i2rt wheel 参数或独立安装步骤。

### 6.2 必需环境变量

以下变量没有默认值，必须设置：

```bash
export YAM_LEFT_GRIPPER_CLOSED_RAD='<left-closed-motor-rad>'
export YAM_LEFT_GRIPPER_OPEN_RAD='<left-open-motor-rad>'
export YAM_RIGHT_GRIPPER_CLOSED_RAD='<right-closed-motor-rad>'
export YAM_RIGHT_GRIPPER_OPEN_RAD='<right-open-motor-rad>'

export YAM_TOP_CAMERA_SERIAL='<top-realsense-serial>'
export YAM_LEFT_CAMERA_SERIAL='<left-realsense-serial>'
export YAM_RIGHT_CAMERA_SERIAL='<right-realsense-serial>'
```

夹爪变量必须按物理语义填写 closed 和 open，不要按数值大小排序。closed 大于 open 完全合法。

以下 CAN 变量可选；不设置时使用右侧默认值：

```bash
export YAM_LEFT_FOLLOWER_CAN='can_left'   # default: can_left
export YAM_RIGHT_FOLLOWER_CAN='can_right' # default: can_right
export YAM_LEFT_LEADER_CAN='can_lead_l'   # default: can_lead_l
export YAM_RIGHT_LEADER_CAN='can_lead_r'  # default: can_lead_r
```

四个解析后的 channel 必须互不相同。单节点采集不要求用户设置 `RLINF_NODE_RANK`；Cluster
会将唯一节点视为 rank 0。

### 6.3 采集命令

在 RLinf 仓库根目录执行：

```bash
bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data
```

YAM 配置默认采集 50 个 `pick_block` 成功 episode。若要长期使用其他任务或数量，请复制或
修改 `realworld_dual_yam_collect_data.yaml` 中的以下字段，再继续用同一启动形式：

```yaml
runner:
  num_data_episodes: 50
env:
  eval:
    override_cfg:
      task_description: pick_block
```

该命令直接进入 RLinf 的：

```text
collect_real_data.py
  -> DataCollector
  -> RealWorldEnv
  -> CollectEpisode
  -> DualYamLeaderIntervention
  -> DualYamJointEnv
```

不存在采集后再运行外部 `--convert` 的步骤；LeRobot 数据由 RLinf 的 `CollectEpisode` 直接写出。

### 6.4 主臂按钮和夹爪语义

两只 leader 的同类按钮按 OR 合并，任意一只手柄都可以触发：

- top button：切换 follower 同步。
  - 开启时先用 `engage()` 按 `engage_duration_s` 和 `max_joint_delta` 平滑对齐 follower；
  - 开启后 leader 关节和手柄夹爪控制 follower；
  - 关闭时 follower 保持实测位置，leader 回到重力补偿 idle。
- record button：控制 episode。
  - reset 后第一次按下：开始记录；
  - 记录中再次按下：以 `reward=1`、`terminated=true`、`manual_done=true` 成功结束并保存。

按钮只响应上升沿并有 0.2 秒 debounce。启动/reset 读取第一帧时应松开两个按钮，否则 idle
极性学习可能把按下状态误当作静止状态。当前没有单独的“失败结束/丢弃”手柄按钮。

teaching handle trigger 默认是：

- 松开：归一化夹爪 1，open；
- 按下：归一化夹爪 0，closed。

将对应 leader 的 `gripper_invert` 设为 `true` 可以反转该映射。

### 6.5 14D 状态和动作 schema

状态和动作固定为：

```text
[L_q0, L_q1, L_q2, L_q3, L_q4, L_q5, L_grip,
 R_q0, R_q1, R_q2, R_q3, R_q4, R_q5, R_grip]
```

| 项 | shape | 单位/语义 |
|---|---:|---|
| follower measured state | `(14,)` | 关节为 rad；夹爪 `0=closed, 1=open` |
| accepted/expert action | `(14,)` | absolute joint target；同一夹爪语义 |
| RGB frame | `(480, 640, 3)` | HWC、`uint8`、RGB |

RLinf 保存的是实测 follower state。主臂接管时保存的是 runtime 安全检查后实际接受的完整 14D
动作，不是 collector 传入的全零 placeholder。

### 6.6 camera key 映射

station 中的源 key 为：

```text
top_rgb, left_rgb, right_rgb
```

当前通用 `RealWorldEnv` 和 `CollectEpisode` 的映射为：

| station/Gym key | `RealWorldEnv` key | 当前 LeRobot feature | 含义 |
|---|---|---|---|
| `top_rgb` | `main_images` | `image` | 主视角；由 `main_image_key: top_rgb` 选择 |
| `left_rgb` | `extra_view_images[:, 0]` | `extra_view_image-0` | 左视角 |
| `right_rgb` | `extra_view_images[:, 1]` | `extra_view_image-1` | 右视角 |

左右顺序来自 `RealWorldEnv` 对剩余 frame key 的字典序排序。当前磁盘 feature 不是
`observation.images.top_rgb` 等 YAM 专名；后续 OpenPI dataconfig 应显式将当前 writer key
映射为模型所需的规范 key，建议：

```text
image              -> base_0_rgb
extra_view_image-0 -> left_wrist_0_rgb
extra_view_image-1 -> right_wrist_0_rgb
```

不要把这份未来 OpenPI 映射误认为当前 LeRobot 数据已经采用这些 feature 名。

### 6.7 准确输出路径

`collect_data.sh` 每次创建：

```text
<RLinf-repo>/logs/YYYYMMDD-HH:MM:SS/
```

默认单 Worker、`resume=false`、`finalize_interval=0` 时输出为：

```text
logs/YYYYMMDD-HH:MM:SS/
├── run_embodiment.log
├── collected_data/
│   └── rank_0/
│       └── id_0/          # LeRobot dataset 与 meta/data/video 文件
└── demos/                 # RLinf TrajectoryReplayBuffer 的 .pt 轨迹
```

- `run_embodiment.log` 第一行记录完整 shell 命令，随后追加 stdout/stderr。
- `collected_data/rank_0/id_0` 是本次直接写出的 LeRobot 数据集。
- `demos` 是 `DataCollector` 同时维护的 RLinf replay trajectory，不应与 LeRobot 目录混淆。
- `finalize_interval=0` 禁止周期性 finalize，所以正常采集期间保持一个 `id_0` shard，并在
  wrapper 干净关闭时生成/完成 metadata。

因此，异常断电或强制杀进程可能留下未 finalize 的 `id_0`。若更看重中途恢复，可改为正的
`finalize_interval`，但每次周期 finalize 会开始新的 `id_N` shard；启用 `resume` 时也应将
`runner.logger.log_path` 固定到希望复用的数据根目录，而不是默认的新时间戳目录。

## 7. 无 YAM 应用仓库依赖与 lazy i2rt 边界

当前目标依赖关系是：

```text
RLinf examples / scheduler / env / collector / LeRobot writer
                             |
                             `-> pinned i2rt SDK（仅真机 connect）
```

明确不需要：

- clone `yam-abc-reproduce` 或其他 YAM 应用 repo；
- 从应用 repo import collector、config、converter 或 runtime；
- 配置应用 repo 路径；
- 采集后运行应用 repo 的 `--convert`。

`rlinf/envs/real/yam/i2rt_backend.py` 是唯一包含 i2rt import 语句的 YAM 模块，而且 import
位于 `_build_yam()` 内。调用顺序是：

```text
Gym/env 构造
  -> I2RTYamBackendFactory（仍未 import i2rt）
  -> 第一次 reset
  -> backend.connect
  -> _build_yam
  -> import i2rt
```

所以 scheduler、actor/rollout 节点、文档工具和 mock 单测不需要安装 i2rt。只有实际拥有 YAM
station 的 env Worker Python 环境需要它。

## 8. 已知限制与后续工作

### 8.1 当前已知限制

- 尚未执行 YAM 真机、相机或 CAN 验收；当前结果来自静态审查和 mock/stub 测试。
- 当前仅支持 14D absolute joint，不支持 TCP/rot6d action。
- 相机当前只支持 RealSense RGB，不支持 depth 或其他 camera backend。
- 完整 station 必须在一个 env Worker 节点可访问，尚无跨节点 YAM Controller Worker。
- scheduler 枚举刻意不探测 CAN 在线，channel/权限/bitrate 错误会在第一次 `reset()` 才出现。
- 官方 i2rt v1.3.3 没有公开每设备数值 `grav_comp_kd` 和 `coulomb_friction` 构造覆盖；RLinf
  不会修改 SDK 私有数组，显式配置而 SDK 不支持时会拒绝启动。
- 当前 i2rt health 检查需要兼容访问 `running`、server thread 和 feedback timestamp；待 SDK
  提供公开 health API 后应替换。
- 真机使用的 i2rt build 仍需包含构造失败时 CAN 回收、控制线程 join 和多圈/有方向 gripper
  limits 修正；外层 adapter 在 SDK 尚未返回 robot handle 前无法完整回收 SDK 内部局部对象。
- `finalize_interval=0` 依赖干净关闭才能完成 metadata，异常退出恢复策略仍需真数据验证。
- 当前通用 LeRobot writer 使用 `image`/`extra_view_image-*`，YAM 到 OpenPI 的命名和 transform
  尚未落地。
- 当前手柄只有“同步切换”和“开始/成功结束”语义，没有独立失败/abort 或左右手独立接管 mask。

### 8.2 安装目标

当前已增加 `requirements/install.sh embodied --env yam`。它已经固定 RLinf 侧的 LeRobot
commit，并通过 `requirements/embodied/envs/yam.txt` 固定官方 i2rt commit；不会下载 YAM
应用仓库，也不接受额外的 i2rt wheel 路径。部署前仍需确认：

- 清单中固定的 i2rt commit 已通过目标机器低速真机验收；
- 与 writer/reader 兼容的 LeRobot 版本；
- RealSense 和 SocketCAN 运行依赖。

部署方仍应保存最终环境 lock、i2rt commit 和校验信息；固定依赖不会替代 SDK 兼容性与
真机验收。

### 8.3 OpenPI 接入目标

至少还需要：

1. 新增 YAM 专用 OpenPI dataconfig/policy transform；
2. 显式映射当前 LeRobot camera feature 到
   `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`；
3. 固定 `state_dim=14`、`action_dim=14` 和 action horizon；
4. 从 YAM 数据重新计算 norm stats，禁止复用 Franka/Aloha stats；
5. 增加 YAM SFT、纯策略 eval 和后续 DAgger example YAML；
6. 在 dataset metadata 中冻结 action representation、单位、夹爪语义和 schema version。

### 8.4 真机验证门禁

建议按以下顺序推进：

1. 校准并复核两只 follower 的 `[closed, open]`，包含降序电机方向测试。
2. 单独验证三台相机的 serial、颜色、尺寸、左右顺序和 stale timeout。
3. 只连接一只 follower，低速验证 startup hold、joint limits、slew、NaN 和 timeout。
4. 验证两只 follower 的 14D slice 不串扰，以及异常后两臂均进入 hold。
5. 分别标定两只 leader 的 `ee_mass`/重力补偿；保持 `bilateral_kp=0` 完成第一轮。
6. 验证 top/record 按钮、防抖、sync engage、episode 结束和 feedback release。
7. 采集 3 条短 episode，检查 `id_0/meta`、14D 无 NaN、三路视频帧数和 task。
8. 用 RLinf/LeRobot loader 做 smoke test，再计算 norm stats 和小数据 overfit。
9. 最后才进行纯策略真机评测和在线 DAgger；policy 超时必须 hold，不能下发零位。

在上述门禁完成前，当前实现应视为“代码和 mock 契约已接通”，而不是“YAM 真机 pipeline 已验收”。
