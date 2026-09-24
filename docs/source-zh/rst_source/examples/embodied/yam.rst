采集双臂 YAM 示教数据
======================

使用 RLinf 原生的 YAM 环境，通过两台电机主臂遥操作两台从臂，并将
14 维关节空间示教直接写成 LeRobot 数据。运行时只依赖 RLinf 和一个固定版本、
经过兼容性验证的 ``i2rt`` 安装包，不需要再下载或导入另一份 YAM 应用仓库。

.. warning::

   本页的数据采集命令会在第一次环境 ``reset()`` 时打开 3 台相机和 4 个
   SocketCAN 设备，并可能驱动两台从臂。运行前请清空工作空间、确保物理急停
   随手可及、松开两个示教手柄的按钮，并让经过培训的操作员留在设备旁。
   只阅读文档或运行下文列出的单元测试不会打开硬件。

PICO VR 数据采集
----------------

第一版 VR 采集使用 ``realworld_dual_yam_collect_data_pico`` 配置，沿用 Franka
的左右两个独立 ``PicoExpert``。手柄末端增量恢复成绝对 TCP 目标后，使用固定版本
i2rt 的 FK/IK 求解关节角；从臂仍接收 14 维绝对关节动作，LeRobot 保存运行时的
``accepted_action``。VR 模式只打开两条 follower CAN 和三路相机，不连接 leader。

当前支持 ``yam + flexible_4310`` 模型。默认 TCP 是 ``grasp_site``，位于模型夹爪
坐标系局部负 Z 方向 100 mm 处，并包含模型安装变换。模型有六个机械臂转动关节和
两个夹爪滑动关节；夹爪归一化位置映射到 XML 范围，在 IK 中固定。运动前需核对模型
TCP 与实际工具。两臂各自持有运动学实例，通过
``pico.left/right.operator_to_robot_yaw`` 分别配置操作者与基座的朝向关系。

YAM 安装入口已包含 pyzmq、Mink、MuJoCo 和 quadprog。外部 publisher 的准备见
:doc:`franka_vr`。先在仓库根目录运行不连接硬件的合成数据检查：

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate
   python -m toolkits.realworld_check.test_yam_vr_pipeline --steps 120

添加 ``--zmq-addr tcp://<publisher-ip>:5555`` 可使用真实 VR 输入。这个工具始终
使用模拟从臂和相机，没有真机执行选项。先释放两只 grip，再握住任一 grip 移动手柄。
报告包含单臂 IK 耗时、含节拍等待的整步耗时及故障次数；检查期间发生故障或始终没有
手柄接管时返回非零退出码。

采集前导出站点相机序列号与 VR 地址，然后运行专用配置：

.. code-block:: bash

   export YAM_TOP_CAMERA_SERIAL=<top-serial>
   export YAM_LEFT_CAMERA_SERIAL=<left-serial>
   export YAM_RIGHT_CAMERA_SERIAL=<right-serial>
   export YAM_PICO_ZMQ_ADDR=tcp://<publisher-ip>:5555
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

该配置保留启动时夹爪自动标定，采集会控制真实从臂。本页站点准备和有人值守的首次
运动验证同样适用。VR 强制启用运行时关节限制，VR 与电机主臂模式同时启用会在打开
硬件前报错。

操作方式：

- 启动、显式 reset、数据失效或运行时故障后，先释放两只 grip，再握住超过 ``0.85`` 接管对应臂。
  接管首步建立新参考，不继承旧运动目标。
- 未接管时用 trigger 重标定。左手 X/Y 关闭/打开左夹爪，右手 A/B 关闭/打开右夹爪；
  没有夹爪按钮时保持原目标。夹爪按钮只在对应手接管时生效。
- 正常未接管时保持固定关节目标：启动释放确认或松开该侧 grip 时记录一次实测关节角，
  后续不随实测漂移更新。重新接管以当前实测位姿建立 VR 参考；数据故障或显式 reset
  清除旧的保持目标，恢复后重新记录。关节限幅仍由运行时执行。
- 右菜单开始录制或成功结束；左菜单丢弃当前片段。按钮可配置，按上升沿触发并消抖。
  正常录制边界不取消健康的遥操作握持状态。
- 可选键盘回退：设置 ``pico.keyboard_enabled: true``，并将
  ``RLINF_KEYBOARD_DEVICE`` 指向有读取权限的 ``/dev/input/by-id/...`` 键盘设备。
  R 开始/结束，X 丢弃，不依赖图形界面。
- 等待录制期间遥操作持续运行但不积累数据帧。IK 的所有尝试失败时清空当前录制缓存，
  本帧只保持失败的一臂，另一臂可继续执行有效目标。保留手柄参考，下一帧自动重试；
  无需松开 grip，失败的关节目标不会下发。
  数据失效、无效手柄动作或运行时故障仍要求释放两只 grip 后重新接管。

示例的平移/旋转增益均为 0.65，将相对接管参考的平移增量和旋转向量乘以 0.65
后生成累计目标。例如手柄移动 10 cm 对应目标移动 6.5 cm，旋转 20° 对应目标旋转 13°。
YAM 使用原有 Franka
``PicoExpert`` 的参考位姿和平移增量，并将裁剪前的完整
累计末端目标交给 IK，避免实测位置滞后时 5 mm 的局部目标导致运动停滞。
``max_position_delta`` 和 ``max_rotation_delta`` 在 YAM 中仅用于动作编码和解码，
不再限制 TCP 目标。Franka 默认的裁剪行为不变。

示例设置 ``pico.ik_backtrack_attempts: 3``：完整目标因不收敛、残差过大或解超出
关节限位而失败时，在本帧依次尝试当前实测 TCP 到目标的 1/2、1/4、1/8 增量。
平移按比例缩小；旋转缩放最短空间旋转的旋转向量，并左乘当前姿态。
各次尝试都从同一实测关节角初始化，不使用失败解作为种子。第一个通过全部校验
的中间目标解进入现有的关节限幅流程，缩步成功不会中断录制。
单臂本帧所有尝试共用 ``pico.ik.max_solve_s`` （示例 30 ms）；超时、无效解、
夹爪关节漂移、种子越界和求解器异常不触发缩步。SDK 调用不可中途抢占，
预算在调用间及返回后检查，超过总预算的结果不下发。这不是硬实时期限。
``left/right_ik_attempts`` 报告包含首次求解的尝试次数，
``left/right_ik_target_fraction`` 报告最后尝试的比例，
``left/right_ik_initial_fault`` 保留首次失败原因；末端残差相对于最后尝试的目标计算。
下一帧重新从实测位姿尝试最新完整 VR 目标，不排队执行旧目标。
此功能减少大目标求解失败，不提供关节速度或加速度约束。
将 ``ik_backtrack_attempts`` 设为 0 可关闭本帧缩步重试。

YAM 默认 ``rotation_delta_frame: operator``，让旋转和平移使用相同的校准操作坐标系。
设 ``C0``、``C`` 是接管时和当前手柄在该坐标系的旋转，``O`` 是操作者到机器人
的 yaw 旋转，则 ``D = O @ (C @ C0.T) @ O.T``，``R_target = D @ R_reference``
（旋转增益不为 1 时先缩放 ``D`` 的旋转向量）。``C`` 已包含原始 VR 轴转换和头显
标定，不能再做一次固定轴映射。这样同一空间旋转轴不会随接管时的握持姿态改变。
``controller_local`` 可显式恢复旧 Franka 的局部增量规则；Franka 默认仍为该规则。
旋转围绕 ``grasp_site`` 保持 TCP 位置，并非单独转动最后一个关节；部分姿态可能无解。

YAM 用 ``rlinf/robotics/parts/teleop/yam_pico.py`` 中的 ``yam_pico`` 设备
（内部为 ``_YamPicoArm`` 与 ``_delta_to_tcp_pose``）
按 ``p_target = p_current + delta_p``、``R_target = delta_R @ R_current`` 构造
目标，参考位姿是接管瞬间实测的 TCP 位姿；YAM 在编码和解码时均关闭运动裁剪。
IK 收敛后，相对实测关节角向通过校验的目标解按统一比例插值。示例通过
``override_cfg.max_joint_delta_per_joint: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]``
将 J1～J6 的每步上限统一为 0.3 rad（左右臂共用）。
整体插值比例取所有关节的 ``limit_i / abs(q_target_i - q_measured_i)`` 与 1 的最小值，
保持各关节运动比例；随后运行时按相同上限和硬关节限位再次检查。
向量设为 ``null`` 时回退到标量 ``max_joint_delta`` （示例 0.08 rad）。
这些值限制每次指令的关节差值，并非末端角速度或加速度限制，效果依赖实际控制周期。
``left/right_joint_step_fraction`` 报告比例，``left/right_limiting_joint`` 报告瓶颈
关节编号（1～6，0 表示未限幅），``left/right_ik_max_joint_delta`` 报告完整解的
最大关节变化量。限幅时每臂最多每秒输出一次诊断日志。
中间关节姿态不保证末端沿直线运动。求解失败、末端残差过大、夹爪模型关节漂移和
关节越界仍会被拒绝；``left/right_ik_fault`` 分别报告失败原因。
不再使用 ``pico.ik.max_joint_delta``。初始关节姿态仍应留出活动余量。
``ik.max_solve_s`` 与 ``max_tick_s`` 是同步计算返回后的过期拒绝阈值，不是可中断的
硬实时截止时间；需在控制机实测整条链路后确认 30 Hz 是否满足要求。

示例开启 ``pico.log_control_timing``，每秒输出 ``YAM VR timing``。
``gap`` 是调用底层关节指令的间隔，``compute`` 是 VR/FK/IK 目标计算耗时，
``ik`` 是双臂 IK 耗时之和，``pace`` 是节拍等待，``send`` 是运行时检查和 SDK
指令调用耗时，``state`` 是观测关节读取耗时，``cameras`` 包含相机等待及图像处理，
``outside`` 是两次 wrapper tick 之间的耗时（含外层采集处理及日志）。这些数值是
窗口内各项最大值，单位 ms，不一定来自同一帧，也不能直接相加。
``rate`` 为窗口内 tick 频率，``fault_ticks`` 为出现故障的 tick 数。
即使相机等待尚未达到超时阈值，也能通过耗时看到；诊断不改变控制目标。

两个订阅器可能读取不同帧，过期判断使用本机接收时间，不能识别 publisher 持续重发
冻结跟踪数据的情况。当前模型不提供双臂互碰或场景避障。本版本用于人工采集，不支持
策略与 VR 的 DAgger 控制权切换。离线测试只验证实现契约，物理标定与真机验收仍需
在实际站点完成。

概览
----

当前示例覆盖 RLinf 原生环境接入和示教数据采集。YAM 专用的策略数据变换、
SFT 和部署配置属于后续集成阶段。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      采集阶段不需要策略模型

   .. grid-item-card:: 算法
      :text-align: center

      主从臂遥操作 · 可扩展为 DAgger 干预

   .. grid-item-card:: 任务
      :text-align: center

      双臂关节空间示教

   .. grid-item-card:: 硬件
      :text-align: center

      2 台从臂 · 2 台电机主臂 · 3 台 RealSense

| **本页流程：** 安装运行环境 → 配置一套 YAM 工作站 → 导出标定值 → 采集成功轨迹 → 检查 RLinf 与 LeRobot 两类输出。
| **前置条件：** :doc:`安装 RLinf <../../start/installation>` · Linux SocketCAN · 3 台 Intel RealSense · 经过验证的 ``i2rt`` build · 物理急停。

RLinf 如何组织机器人设备
-------------------------

RLinf 把“资源调度”和“设备控制”明确分开：

.. code-block:: text

   Hydra YAML
     -> NodeHardwareConfig(type="DualYam")
     -> DualYamConfig
     -> DualYamDiscovery.enumerate()         # 只处理配置
     -> 每套完整工作站对应一个 RobotInfo
     -> component_placement 选择 hardware rank 0
     -> WorkerInfo.hardware_infos
     -> RealWorldEnv._create_env()
     -> create_DualYamJointEnv()
     -> 可选 DualYamLeaderIntervention
     -> 可选 YamPico 设备 + YamPicoEpisode    # VR 采集封装
     -> DualYamJointEnv
     -> YamControlRuntime                    # 从臂命令的唯一写入者
     -> 延迟加载 i2rt backend

一份 ``DualYamConfig`` 代表一套完整工作站，其中同时包含左右从臂、左右主臂和
所有相机。调度器把它枚举成一个 hardware rank。因此 ``placement: 0`` 表示
“第一套完整 YAM 工作站”，并不是 node rank 0，也不是 ``can0`` 接口。node rank
与 hardware rank 的区别请参见 :doc:`资源放置 <../../concepts/placement>`。

调度阶段只校验配置和资源独占关系，不导入 ``i2rt``、不探测 CAN、也不打开
相机。环境 worker 被分配到资源后，才通过 ``WorkerInfo.hardware_infos`` 收到
描述该工作站的 ``RobotInfo``，task factory 再从 ``robot_info.config`` 取到同一个
``DualYamConfig``。真正的硬件连接推迟到第一次 ``reset()``：

.. code-block:: text

   打开并预热全部相机 -> 连接左从臂 -> 原位保持
                    -> 连接右从臂 -> 原位保持 -> 校验反馈
                    -> 仅在启用主臂干预时连接两台主臂

纯策略推理时关闭主臂干预，因此不会打开两个主臂 CAN。采集配置则会启用主臂
wrapper。当前实现要求 4 个 CAN 接口和所有相机都能被同一个环境 worker
所在节点直接访问。

配置边界
~~~~~~~~

三类配置应分别放在对应的 YAML 区域：

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - 配置类别
     - 所在位置
     - 负责内容
   * - 物理工作站
     - ``cluster.node_groups[].hardware``
     - 节点归属、CAN 名称、机械臂/夹爪型号、各设备质量和补偿参数、夹爪原始端点、相机序列号。
   * - 任务与运行时
     - ``env.eval.override_cfg``
     - 任务文本、控制频率、RLinf 关节/步进限制、超时、图像尺寸和主臂回合控制行为。
   * - 进程放置
     - ``cluster.component_placement``
     - 哪个 worker 独占接收一整套工作站资源。

安装
----

通过 RLinf 既有安装入口创建 YAM embodied 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

该入口会安装 RLinf embodied 依赖、RealSense/OpenCV、LeRobot，以及
``requirements/embodied/envs/yam.txt`` 中固定 commit 的官方 ``i2rt`` SDK；不要求也不会
clone YAM 应用仓库，同时没有额外的 wheel 环境变量或独立 SDK 安装路径。模块仍然延迟
导入 ``i2rt``，因此仅调度和 dummy 环境的 import 不会触碰硬件。

兼容 build 必须支持 ``get_yam_robot()`` 使用的公开参数，并能可靠停止自身的
CAN/控制线程。如果 SDK 构造函数不支持逐设备数值型阻尼或摩擦力覆盖，适配层会
明确拒绝该配置，不会修改 SDK 私有数组。

启动 RLinf 前，请通过系统常规配置为 4 个 USB-CAN 适配器设置持久 SocketCAN
名称并拉起接口。示例默认值如下：

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - 角色
     - 默认接口
     - 可选环境变量
   * - 左从臂
     - ``can_left``
     - ``YAM_LEFT_FOLLOWER_CAN``
   * - 右从臂
     - ``can_right``
     - ``YAM_RIGHT_FOLLOWER_CAN``
   * - 左主臂
     - ``can_lead_l``
     - ``YAM_LEFT_LEADER_CAN``
   * - 右主臂
     - ``can_lead_r``
     - ``YAM_RIGHT_LEADER_CAN``

以下命令只读取接口状态，不会向机器人发控制命令：

.. code-block:: bash

   ip -details link show can_left
   ip -details link show can_right
   ip -details link show can_lead_l
   ip -details link show can_lead_r

配置工作站
----------

可复用的环境默认值位于
``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``，完整的单节点
采集示例位于
``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``。

采集示例将两个 follower 的 ``gripper_limits`` 设为 ``null``，使用官方 i2rt
自动标定流程。每次启动时，两只夹爪都会分别向两个方向运动，以检测当前编码圈中的
``[闭合, 张开]`` 电机范围。标定完成前必须保证两只夹爪完全无遮挡。

如果某个工作站明确选择跳过启动标定，应将 ``null`` 替换为本工作站的实测值，
顺序固定为 ``[闭合, 张开]``：

.. code-block:: bash

   # 从臂夹爪原始电机位置，顺序必须是 [闭合, 张开]。
   # 数值可以递增，也可以递减；不要排序。
   export YAM_LEFT_GRIPPER_CLOSED_RAD=<左夹爪闭合实测值>
   export YAM_LEFT_GRIPPER_OPEN_RAD=<左夹爪张开实测值>
   export YAM_RIGHT_GRIPPER_CLOSED_RAD=<右夹爪闭合实测值>
   export YAM_RIGHT_GRIPPER_OPEN_RAD=<右夹爪张开实测值>

   # RealSense 序列号。
   export YAM_TOP_CAMERA_SERIAL=<顶部相机序列号>
   export YAM_LEFT_CAMERA_SERIAL=<左侧相机序列号>
   export YAM_RIGHT_CAMERA_SERIAL=<右侧相机序列号>

不要估算固定夹爪端点。这些值是电机弧度原始端点，不是动作中的归一化 ``[0, 1]``。
顺序本身包含电机方向信息，因此合法的 ``[闭合, 张开]`` 数对也可能递减。固定的多圈
限位还必须匹配本次启动的编码圈；除非所安装的 i2rt build 会将持久化限位对齐到当前
编码圈，否则应优先使用自动标定。

如果 CAN 名称不同，再导出上表中的 4 个可选变量。最终解析出的 4 个名称必须
互不相同，相机名称和序列号也必须分别唯一。

各设备调参项与设备本身放在同一段 hardware 配置中：

.. code-block:: yaml

   left_leader:
     ee_mass: null                 # 使用固定 i2rt 模型中的值
     gravity_comp_factor: null     # 使用所选机械臂模型的默认值
     grav_comp_kd: null
     coulomb_friction: null
     use_coulomb_friction: false
     bilateral_kp: 0.0
     gripper_invert: false

``ee_mass`` 和 ``gravity_comp_factor`` 影响重力支撑；``grav_comp_kd`` 是重力补偿
阻尼；``coulomb_friction`` 与 ``use_coulomb_friction`` 控制库仑摩擦补偿。

当前 VR 示例为左右从臂显式设置
``gravity_comp_factor: [1.3, 1.3, 1.3, 1.3, 1.0, 1.0]``，并开启摩擦补偿。
2026-09-08 的本机调参还将已安装 i2rt 的 ``robots/config/yam_v1.yml`` 中
``coulomb_friction`` 设为 ``[0.3, 0.3, 0.3, 0.06, 0.06, 0.06]``。
该 SDK 本地修改会影响使用此模型默认值的程序，不包含在 RLinf 源码依赖中；
重装 i2rt 后需重新核对。更改后重启控制程序以重新加载配置。
``bilateral_kp`` 则控制主臂向从臂实测位置反馈的强度，不是重力补偿参数，初次
采集应保持 ``0.0``。可选值为 ``null`` 时保留固定 SDK 模型的配置。数值型
``grav_comp_kd`` 和 ``coulomb_friction`` 只有在 i2rt 构造函数明确支持时才能使用。

基础配置中的关节上下限是 YAM v1 名义范围。若台面、任务或安装空间更小，应替换
成经过验证的更窄工作区。切换其他 YAM 型号时，必须同时修改 ``arm_type`` 和
RLinf 关节限制；如果 RLinf 配置超出 SDK 限制，启动检查会拒绝继续。

配置初始位置
~~~~~~~~~~~~

可复用环境 YAML 中包含一个默认关闭的 ``reset`` 配置块。先把 follower 移到经过
验证、不会碰撞的初始姿态，记录 follower 的 14 维实测状态，再分别填入左右两组
7 维数值；不要使用未经核对的 leader 位置。每组最后一个值是归一化夹爪位置
（``0=闭合``，``1=张开``）：

.. code-block:: yaml

   reset:
     enabled: true
     mode: startup
     left_qpos:  [q0, q1, q2, q3, q4, q5, 1.0]
     right_qpos: [q0, q1, q2, q3, q4, q5, 1.0]
     duration_s: 4.0
     max_joint_delta: 0.05
     tolerance: 0.03
     timeout_s: 8.0

数采使用 ``startup``：采集器虽然会在每个片段结束后调用 ``reset()``，但 follower
只会在进程首次启动时归位。纯策略部署使用 ``episode``，每个 episode 开始前恢复
同一个初始位置。``manual`` 只在调用
``reset(options={"reset_qpos": True})`` 或基础环境的
``env.unwrapped.reset_to_configured_qpos()`` 时运动。

归位过程会按 ``step_frequency`` 做平滑关节空间插值，始终检查硬关节限位，失败时
保持当前位置，并验证最终实测误差。关节空间直线路径不具备避碰能力，只能在验证
所有允许起始区域到目标的整条路径后开启。如果归位前 leader 正处于同步状态，系统
会先释放同步所有权；操作者对齐 leader 后需要重新接管。

无硬件验证
----------

正式上机前运行 YAM 单元测试。它们全部使用 mock，不会打开 CAN 或相机：

.. code-block:: bash

   pytest -q \
     tests/unit_tests/test_yam_hardware.py \
     tests/unit_tests/test_yam_runtime.py \
     tests/unit_tests/test_yam_env.py \
     tests/unit_tests/test_yam_intervention.py \
     tests/unit_tests/test_yam_imports.py \
     tests/unit_tests/test_yam_examples.py

这些测试覆盖注册与配置转换、14 维顺序、关节/步进限制、陈旧或非有限反馈处理、
干预所有权、清理以及 ``i2rt`` 延迟导入，但不能替代低速真机验收。

采集示教
--------

.. danger::

   下一条命令是真机启动点。第一次 ``reset()`` 会打开相机、连接两台从臂，随后
   连接两台主臂。不要把它当成只检查配置的命令运行。

在 RLinf 仓库根目录启动默认 50 回合采集：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
     realworld_dual_yam_collect_data

示例配置默认采集 50 个回合，任务串是一条整桌整理指令。为了保持 RLinf 既有的“配置名启动”
风格，如需创建其他任务配方，请复制或修改
``realworld_dual_yam_collect_data.yaml`` 中的以下字段：

.. code-block:: yaml

   runner:
     num_data_episodes: 50
   env:
     eval:
       override_cfg:
         task_description: "Tidy up the table. ..."   # 当前配置的指令

该流程会：

1. 调用 RLinf 通用的 ``collect_real_data.py`` 入口；
2. 让调度器分配一份完整 ``DualYam`` 资源；
3. 构建 ``RealWorldEnv`` 和已注册的 ``DualYamJointEnv-v1``；
4. 启用电机主臂干预与按钮回合控制；
5. 将录制的回合直接写入 LeRobot，跳过 RLinf replay buffer。

整个过程没有 ``--convert`` 阶段，也不会在运行时 clone 或 import YAM 应用仓库。

示教手柄按钮
~~~~~~~~~~~~

连接主臂时先松开两个按钮。首个采样会建立每个手柄的空闲电平；任一手柄上的
按钮都会控制整套双臂工作站。

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - 控件
     - 当前状态
     - 结果
   * - 顶部/第一个按钮
     - 未同步
     - 从从臂实测位置平滑过渡到当前主臂位置，随后两台主臂共同控制两台从臂。
   * - 顶部/第一个按钮
     - 已同步
     - 两台从臂原位保持，两台主臂回到重力补偿空闲状态。
   * - 录制/第二个按钮
     - 回合开始前等待
     - 从当前位置开始一个新录制回合。
   * - 录制/第二个按钮
     - 正在录制
     - 以 reward ``1`` 和 success 结束回合，并保持遥操同步以继续下一个回合。
   * - 夹爪扳机
     - 默认映射
     - 松开为夹爪 ``1`` （张开），按下为 ``0`` （闭合）；在对应主臂设置 ``gripper_invert: true`` 可反转。

示例使用 ``sync_on_reset: false``，操作者准备好后只需按一次顶部按钮接管；
``preserve_sync_between_episodes: true`` 使录制按钮只负责切分回合，不释放遥操，
只有顶部按钮切换同步状态。同时使用 ``unsynced_action_source: hold``，所以
collector 的占位零动作不会把从臂送向零位。按钮事件采用上升沿触发并做消抖。

终端仅在切换时提示 ``[待录制]``、``[录制中]`` 和 ``[录制结束]``。
后台保存完成不另行提示；进度条仅计数，不代表落盘完成。

主臂采集默认 ``data_collection.export_mp4: true``。每条回合落盘后，独立低优先级
进程按视角顺序导出 H.264 MP4，不等待整次采集结束，也不阻塞录制主循环。
输出为 ``collected_data/rank_0/review_videos/id_N/episode_XXXXXX/`` 下的
``top.mp4``、``left.mp4``、``right.mp4``；不另报导出开始或完成，失败仍会提示。
编码需要时间，若尚未完成即可继续录下一条，任务按顺序排队；正常退出先释放
机器人接口，再等待视频收尾。MP4 用于视觉验收，训练仍读取原始 Parquet。

观测与动作契约
--------------

所有状态和动作都使用同一绝对 14 维布局：

.. code-block:: text

   [left_q0, ..., left_q5, left_gripper,
    right_q0, ..., right_q5, right_gripper]

机械臂关节单位为弧度；夹爪归一化为 ``0=闭合, 1=张开``。观测中的关节值来自
从臂实测位置。所有命令都会校验精确 shape 和有限性，并把夹爪裁剪到有效范围。
默认情况下，机械臂命令还会裁剪到配置限位，并根据实测位置应用
``max_joint_delta`` 步进限制。采集示例设置
``enforce_runtime_joint_limits: false`` 来对齐旧版 yam-abc 遥操路径：平滑接合完成后，
主臂关节目标直接交给 i2rt，由 i2rt 应用硬件限位。wrapper 会把最终目标写入
``intervene_action``。

相机与数据键会在三层边界发生有意的重命名：

.. list-table::
   :header-rows: 1
   :widths: 23 37 40

   * - 边界
     - 键名
     - 含义
   * - YAM Gym 观测
     - ``frames.top_rgb``、``frames.left_rgb``、``frames.right_rgb``
     - 具名 RGB 帧；``state.joint_position`` 是 14 维实测状态。
   * - ``RealWorldEnv``
     - ``main_images``、``extra_view_images``
     - ``top_rgb`` 成为主视角；其他名称排序后，当前配置中 index 0 是 ``left_rgb``，index 1 是 ``right_rgb``。
   * - RLinf LeRobot writer
     - ``image``、``extra_view_image-0``、``extra_view_image-1``
     - 最终数据集中顶部、左侧、右侧相机的字面 feature 名；状态和动作分别是 ``state``、``actions``。

如果下游 transform 依赖该顺序，请保留示例中的相机名称。通用 collector 会保留
视角顺序，但不会在最终 LeRobot 列中保留 ``left_rgb``/``right_rgb`` 语义名。
后续 YAM 策略 dataconfig 需要显式映射这些字面键名。

输出目录
--------

``collect_data.sh`` 会创建新的 ``logs/<timestamp>/``。成功回合写入其下的
LeRobot 数据集：

.. code-block:: text

   logs/<timestamp>/
   `-- collected_data/
       `-- rank_0/
           `-- id_0/                 # 本次运行的 LeRobot shard
               |-- meta/info.json
               |-- meta/episodes.jsonl
               |-- meta/tasks.jsonl
               |-- meta/stats.json
               |-- data/...
               |-- videos/...        # 具体布局取决于 LeRobot 版本
               |-- recording_errors.jsonl          # 仅在发生溢出后出现
               `-- invalid_episodes/overflow_XXXX/ # 一次溢出的截断前缀
                   |-- frames.pkl
                   `-- <相机键名>/

示例开启 ``streaming: true`` 并关闭 ``runner.save_demos``：每一帧在录制时直接
写入 LeRobot。LeRobot v2 的无损 PNG 由现有后台线程直接写入，关闭压缩以降低 CPU 开销，
不再启动第二层图像队列。主循环不会因写入队列满而等待：若积压达到 240 个任务，
停止当前回合的录制并报告 ``recording_invalid``，遥操继续。截断前缀的图像与
``frames.pkl`` 被移入新建的 ``invalid_episodes/overflow_XXXX/`` 目录，原因追加到
shard 根目录的 ``recording_errors.jsonl``；它不进入正常 LeRobot 元数据，也不计成功。待写入追上后
结束当前录制、重新开始。PNG 不压缩会增加临时磁盘占用和写入带宽。
LeRobot v2 在回合结束时每批嵌入 16 帧图像并写入 Parquet，完成后再发布文件；
流式模式每条保留回合使用独立 ``id_N`` 分片，由独立后台线程串行保存、释放资源。
新回合立即向新的分片写帧，不等待上一条 Parquet 封装完成。
MP4 按单个 Parquet 行组分批读取，避免跨行组迭代持续保留图像缓冲。
不会把已保存回合合并保留在 ``hf_dataset`` 内存中。当前回合的状态、动作、
图像路径和索引仍暂存内存，因此不能将整个采集进程视为严格恒定内存。
成功和失败的回合都会保存，回合级 ``is_success`` 在回合结束时统一盖写，训练时
可按该标志过滤。同时不再生成 ``demos/`` 目录。``finalize_interval`` 仅影响
非流式模式；流式模式每条保存后释放 writer。LeRobot v2 的回合元数据和 Parquet
在 ``save_episode`` 成功返回时已保存；成功按钮只提交保存任务，不能作为落盘完成的凭据。
只有复用显式 ``save_dir`` 时才应设置 ``resume: true``；恢复
运行会写新的 ``id_N`` shard，不覆盖已 finalize 的 shard。

LeRobot 帧包含 ``state``、``actions``、``image``、``extra_view_image-0``、
``extra_view_image-1``、``done``、``is_success``、``intervene_flag`` 和
``segment_id``。任务文本通过 LeRobot task metadata 保存。通用 writer 行为请参见
:doc:`数据采集 <../../guides/data_collection>`。

YAM 文件职责总览
----------------

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - 文件
     - 含义与职责
   * - ``rlinf/robotics/robots/dual_yam.py``
     - 定义一套完整工作站资源，转换嵌套 Hydra 配置，校验 CAN/相机唯一性，并在不访问硬件的前提下注册机器人类型。
   * - ``rlinf/envs/real/yam/types.py``
     - 定义统一 14 维状态/动作契约、类型化状态、命令结果和 backend 协议。
   * - ``rlinf/envs/real/yam/config.py``
     - 校验任务层控制频率、关节限制、相机超时和主臂干预行为。
   * - ``rlinf/envs/real/yam/i2rt_backend.py``
     - 作为唯一 ``i2rt`` 边界进行延迟导入，并适配从臂、主臂、按钮、健康状态与清理 API。
   * - ``rlinf/envs/real/yam/mock_backend.py``
     - 为 dummy 模式和测试提供完全不访问硬件的主从臂实现。
   * - ``rlinf/envs/real/yam/control_runtime.py``
     - 统一拥有所有 transport、串行化从臂写入、校验每条命令、平滑接管、故障保持并按安全顺序关闭。
   * - ``rlinf/envs/real/yam/dual_yam_joint_env.py``
     - 实现 Gym action/observation space、延迟启动、相机处理、step 节拍和资源关闭。
   * - ``rlinf/envs/real/yam/leader_intervention.py``
     - 实现双主臂同步、按钮回合控制、policy/hold/leader 命令所有权和 ``intervene_action`` 上报。
   * - ``rlinf/envs/real/yam/__init__.py``
     - 汇总公开 YAM API，并向任务注册表登记 ``DualYamJointEnv-v1``。
   * - ``rlinf/envs/real/yam/pico_episode.py``
     - VR 采集的录制、丢弃与键盘回合控制，即 ``YamPicoEpisode``。
   * - ``rlinf/robotics/parts/teleop/yam_pico.py``
     - ``yam_pico`` 遥操作设备：PICO 读数到 14 维关节目标的映射、IK 与故障保持。
   * - ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``
     - 可复用的 Gym/任务默认值及显式 RLinf 安全限制。
   * - ``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``
     - 一套完整工作站的调度、遥操作和直接 LeRobot 采集配置。
   * - ``requirements/install.sh`` 中的 ``--env yam``
     - 构建包含相机、LeRobot 和固定 i2rt SDK 的完整 YAM 环境，不要求 YAM 应用仓库。
   * - ``requirements/embodied/envs/yam.txt``
     - 固定官方 i2rt commit，并声明原生 runtime 所需的相机和配置依赖。
   * - ``requirements/embodied/envs/yam-build-constraints.txt``
     - 将 i2rt 的 ruckig 源码构建约束限制在 YAM 环境内部。
   * - ``examples/embodiment/collect_data.sh``
     - 所有真机配置共用的通用采集入口；它按配置名启动并创建带时间戳的日志目录。YAM 侧改动只补充了入口文件和日志文件变量，默认日志路径保持通用。
   * - ``rlinf/envs/real/__init__.py``
     - 从 RLinf 真机入口导入 YAM task 包，使 Gym 注册生效。
   * - ``rlinf/robotics/robots/__init__.py``、``rlinf/robotics/__init__.py``、``rlinf/envs/real/__init__.py``
     - 加载 YAM 机器人模块，使 ``DualYam`` 进入 hardware-policy registry；并导入 task 包，使指定 ``DualYam`` 的集群配置能够解析。
   * - ``tests/unit_tests/test_yam_hardware.py``
     - 测试 registry 转换、工作站枚举、资源冲突以及保持方向的夹爪标定。
   * - ``tests/unit_tests/test_yam_runtime.py``
     - 测试延迟连接、命令安全、反馈超时、角色相关 i2rt 模式和清理。
   * - ``tests/unit_tests/test_yam_env.py``
     - 测试 dummy Gym 契约、14 维顺序与幂等关闭。
   * - ``tests/unit_tests/test_yam_intervention.py``
     - 测试策略/主臂所有权、同步失败清理和回合结束行为。
   * - ``tests/unit_tests/test_yam_imports.py``
     - 守护无硬件导入路径，确保 ``i2rt`` 保持延迟加载。
   * - ``tests/unit_tests/test_yam_examples.py``
     - 守护公开 YAML 契约、直接 LeRobot 设置和“环境内固定 SDK、不依赖应用仓库”的安装规则。

已知边界
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 范围
     - 当前边界
   * - 控制表示
     - 目前只有绝对 14 维关节空间；TCP/笛卡尔动作需要另建带版本的环境和数据契约。
   * - 相机
     - 只支持 RealSense RGB，``enable_depth`` 必须为 ``false``。
   * - 放置
     - 从臂、电机主臂和相机必须与同一个环境 worker 共节点；尚未实现按机械臂拆分的远程 controller。
   * - 干预粒度
     - 同步和干预同时作用于双臂，尚无左右独立 intervention mask。
   * - 模型
     - 当前示例只采数据，尚未提供 YAM 专用模型 dataconfig、归一化统计、SFT recipe 和策略部署示例。
   * - SDK 调参
     - 重力系数和摩擦补偿开关使用 SDK 公开参数；逐设备数值型阻尼/摩擦覆盖需要兼容的 ``i2rt`` 构造函数，否则会被拒绝。
   * - 故障响应
     - 软件仅提供尽力而为的实测位置保持和资源清理，不能代替物理急停。
   * - 验证范围
     - 单元测试使用 mock；正式双臂采集前仍需固定 SDK 版本，并逐臂、低速完成真机验收。

本机三视角预览
--------------

主臂采集配置启用 ``env.eval.override_cfg.camera_preview_port: 8080``。
在 yambox 显示屏的浏览器中打开 ``http://127.0.0.1:8080``，即可并排查看
``top_rgb``、``left_rgb``、``right_rgb``。等待录制按钮和录制期间均刷新。
图像来自实际采集的 RGB 观测，仅预览缩小显示，不改变数据集分辨率，
也不会重复打开 RealSense。页面约每 100 ms 请求一次最新画面；
图上的时间表示观测发布后的时间，不是相机硬件曝光时间。

显示服务仅监听本机。关闭网页不影响遥操或录制；正常关闭环境时服务退出。
将该配置设为 ``null`` 可关闭预览。修改配置或代码后需要正常结束当前
采集再重新启动；不要同时启动第二个采集进程。

脚踏板确认保存
~~~~~~~~~~~~~~

主臂采集通过 ``leader_intervention.foot_switch_device`` 指定脚踏板的稳定
``/dev/input/by-id/`` 路径；设为 ``null`` 可恢复手柄结束后直接保存。
已实测 PCsensor 三格脚踏板：左侧 ``KEY_A`` （30）删除，右侧 ``KEY_C`` （46）保存，
本机中间踏板为 ``KEY_B`` （48），已启用返回操作者确认的初始位。手柄录制键仍负责开始/结束，顶部同步键仍控制主从同步。

确认数采初始位后，将 ``reset`` 配置为 ``enabled: true, mode: manual``，
填写同步状态下确认的主臂 ``left_qpos/right_qpos``，并将
``leader_intervention.foot_switch_reset_key`` 设为实测中间踏板键值。
中间踏板在待录制和录制中逐帧驱动主从四臂返回相同绝对关节目标；
录制中的回位过程保留在当前条，回位完成后再按白键结束。夹爪仍由手柄控制。
黄色键可中止回位，回位结束或中止后恢复原主臂重力补偿。
此初始位不是编码器零点；启动及回合结束均不自动回位。
首次需空手、低速验证整段路径，关节插值不包含碰撞规划。


每条结束后显示 ``[待选择]`` 并暂计入进度；此时不再记录数据，仍可遥操。
右脚确认后正式保存并后台导出三视角 MP4，左脚删除当前未确认条的临时图像和状态记录、
回退进度，不改动之前保留的条目。选择后按手柄录制键开始下一条。
长踩自动重复、进入待选择前的旧按键、中间踏板不会保存或删除下一条。
正常关闭或提前 reset 会清理尚未确认的回合。最后一条也必须确认后才结束采集。

脚踏板需 evdev 读取权限。YAM Box 的规则为
``/etc/udev/rules.d/70-yam-foot-switch.rules``：

.. code-block:: text

   SUBSYSTEM=="input", KERNEL=="event*", ATTRS{idVendor}=="3553", ATTRS{idProduct}=="b001", GROUP="plugdev", MODE="0660"

采集用户需在 ``plugdev`` 组；加载规则并重新插入脚踏板后生效。
只独占配置指定的脚踏板，不读取或独占操作员的普通键盘。
