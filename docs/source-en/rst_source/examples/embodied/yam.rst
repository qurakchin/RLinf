Collect Dual-Arm YAM Demonstrations
===================================

Use RLinf's native YAM environment to teleoperate two follower arms from two
motorized leader arms and write 14-D joint-space demonstrations directly in
LeRobot format. The runtime depends on RLinf plus a pinned, compatible ``i2rt``
build; it does not require a checkout of a separate YAM application repository.

.. warning::

   The motorized-leader collection command opens three cameras and four SocketCAN
   devices on the first environment reset, and it can move both follower arms.
   Clear the workspace, make the physical emergency stop reachable, release both
   teaching-handle buttons, and keep a trained operator beside the station.
   Reading this page or running the unit-test command below does not open hardware.

.. _yam-pico-collection:

PICO VR Collection
------------------

The first VR implementation reuses two independent ``PicoExpert`` subscribers,
just like Franka. Each controller's normalized TCP delta is converted into an
absolute target and solved with the pinned i2rt FK/IK implementation. Followers
still receive the existing 14-D absolute joint action; LeRobot stores the
runtime's ``accepted_action``. The leader CAN devices are never connected.

This version supports ``arm_type: yam`` with ``gripper_type: flexible_4310``.
The model uses ``grasp_site`` (100 mm along local negative Z from the gripper
frame, with its model mounting transform), six arm hinges, and two finger
slides. Both slides are mapped from normalized opening to their XML ranges and
frozen during IK. Verify the model TCP against your installed tool before motion.
Each arm has an independent kinematics instance and its own
``pico.left/right.operator_to_robot_yaw`` setting.

Install the normal YAM environment; its requirements include pyzmq, Mink,
MuJoCo and quadprog. Prepare the external publisher following :doc:`franka_vr`.
Then run the hardware-free synthetic check from the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate
   python -m toolkits.realworld_check.test_yam_vr_pipeline --steps 120

To feed live VR through the same check, add
``--zmq-addr tcp://<publisher-ip>:5555``. This diagnostic always uses mock
followers and cameras; it has no hardware execution option. Release both grips
once, then hold either grip and move the controller. The report includes per-arm
IK timing, paced step timing and fault counts. It exits nonzero if faults occurred
or neither hand was active during the observation window.

For collection, export the station's three camera serial variables and the VR
address, then launch the dedicated recipe:

.. code-block:: bash

   export YAM_TOP_CAMERA_SERIAL=<top-serial>
   export YAM_LEFT_CAMERA_SERIAL=<left-serial>
   export YAM_RIGHT_CAMERA_SERIAL=<right-serial>
   export YAM_PICO_ZMQ_ADDR=tcp://<publisher-ip>:5555
   bash examples/embodiment/collect_data.sh realworld_dual_yam_collect_data_pico

The recipe retains startup gripper auto-calibration. Its followers can move on
reset/teleoperation; the existing station setup and supervised first-motion
checks on this page apply. Runtime joint limits are mandatory for VR, and enabling
VR and motorized leaders together is rejected before hardware opens.

Controls:

- Release both grips after startup, explicit reset, data loss or runtime fault, then hold a grip
  above ``0.85`` to control its arm. First engagement records a new reference.
- Normal inactive control holds a fixed joint target captured at startup grip
  release confirmation or when that arm's grip is released. Feedback drift does
  not update this target. Re-engagement uses the current measured TCP pose;
  data faults and explicit resets clear old hold targets before recovery.
  Runtime joint bounds still apply.
- Trigger recalibrates operator heading while inactive. X/Y close/open the left
  gripper; A/B close/open the right. No button preserves the existing target.
  Gripper buttons are effective only while that hand owns control.
- Right menu starts recording or finishes it successfully; left menu discards
  the current recording. These buttons are configurable and edge-debounced.
  A recording boundary does not release an otherwise healthy teleoperation grip.
- Optional keyboard fallback: set ``pico.keyboard_enabled: true`` and
  ``RLINF_KEYBOARD_DEVICE`` to a readable ``/dev/input/by-id/...`` keyboard.
  Press R to start/finish and X to discard. It works without a display server.
- Waiting for recording keeps teleoperation ticking without accumulating frames.
  Failure of all IK attempts clears the recording buffer and holds only the
  failed arm; the other arm can execute its valid target. Hand references are
  preserved, retrying automatically on the next frame without
  releasing either grip. Failed joint targets are never dispatched. Data loss,
  invalid controller actions and runtime faults still require releasing both grips
  before taking control again.

The recipe uses 0.65 translation and rotation gains, scaling displacement and
the rotation vector relative to the grip reference before composing the target.
A 10 cm controller displacement produces a 6.5 cm target displacement, and a
20-degree controller rotation produces a 13-degree target rotation.
YAM reuses the original Franka ``PicoExpert`` references and translation delta.
It passes the full cumulative
TCP target to IK before Cartesian clipping, avoiding stalled motion when a 5 mm
local target stays close to lagging measured feedback. ``max_position_delta`` and
``max_rotation_delta`` only encode/decode actions in YAM; they no longer cap its
TCP target. Franka retains its default clipping behavior.

The recipe sets ``pico.ik_backtrack_attempts: 3``. If the full target fails due
to nonconvergence, excessive residual, or a solution outside joint limits, the
same tick retries 1/2, 1/4, then 1/8 of the delta from the measured TCP pose.
Translation is scaled linearly; rotation scales the shortest spatial rotation
vector and left-multiplies the measured orientation. Every attempt uses the same
measured joint seed, never a failed iterate. The first validated solution enters
the existing joint limiter; a successful retry keeps recording active.
All attempts for one arm share ``pico.ik.max_solve_s`` (30 ms in the recipe).
Timeouts, invalid solutions, finger motion, invalid seeds, and solver exceptions
do not trigger retries. The synchronous SDK call cannot be preempted: time is
checked between calls and on return, and results exceeding the total budget are
not dispatched. This is not a hard real-time deadline.
``left/right_ik_attempts`` includes the initial solve;
``left/right_ik_target_fraction`` reports the last attempted fraction;
``left/right_ik_initial_fault`` preserves the first failure reason. Pose residuals
refer to the last attempted target. The next tick tries the latest full VR target
from measured feedback; old targets are not queued. Backtracking does not add
joint velocity or acceleration constraints. Set ``ik_backtrack_attempts: 0`` to
disable retries within the same tick.

YAM defaults to ``rotation_delta_frame: operator``, using the same calibrated
operator axes for rotation and translation. Let ``C0`` and ``C`` be reference and
current controller rotations in that frame and ``O`` the operator-to-robot yaw.
Then ``D = O @ (C @ C0.T) @ O.T`` and ``R_target = D @ R_reference``; non-unit
rotation gains scale the rotvec of ``D`` first. ``C`` already includes raw VR axis
conversion and head calibration, so do not apply the fixed axis mapping again.
A spatial rotation axis therefore stays independent of the initial hand pose.
Explicit ``controller_local`` restores the original Franka convention, which
remains the Franka default. Rotation holds the ``grasp_site`` position fixed;
it is not a command to rotate only the last joint and can be unreachable.

YAM composes the target in the ``yam_pico`` device
(``rlinf/robotics/parts/teleop/yam_pico.py``, via ``_YamPicoArm`` and
``_delta_to_tcp_pose``): ``p_target = p_current + delta_p``
and ``R_target = delta_R @ R_current``, anchored at the measured TCP pose of the
grip edge. YAM disables motion clipping in both encoding and decoding. After IK
converges, joints interpolate from measured positions toward the validated
solution with one common fraction. The recipe uses
``override_cfg.max_joint_delta_per_joint: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]``
for equal J1..J6 bounds of 0.3 rad on both arms. The common fraction is the
minimum of 1 and all ``limit_i / abs(q_target_i - q_measured_i)`` ratios, retaining
coordinated joint motion. The runtime applies the same bounds and hard joint
limits again. Set the vector to ``null`` to restore scalar ``max_joint_delta``
(0.08 rad in the recipe). These are per-command joint displacement bounds, not
TCP angular velocity or acceleration limits; their effect depends on loop timing.
``left/right_joint_step_fraction`` reports the fraction, ``left/right_limiting_joint``
the bottleneck joint (1..6, or 0 when not limited), and ``left/right_ik_max_joint_delta``
the full solution's largest joint change. Limiting diagnostics log at most once
per second per arm. Intermediate joint poses need
not follow a straight TCP path. IK failures, excessive residuals, finger drift
and joint limit violations are still rejected; ``left/right_ik_fault`` reports
each arm's failure. ``pico.ik.max_joint_delta`` is no longer used.
The initial joint pose still needs room to move.
``ik.max_solve_s`` and ``max_tick_s`` reject late results after synchronous
computation returns; they are not hard real-time deadlines. Measure the full
loop on the robot computer before relying on 30 Hz operation.

The recipe enables ``pico.log_control_timing`` and emits ``YAM VR timing``
once per second. ``gap`` measures time between runtime command calls;
``compute`` includes VR/FK/IK target generation; ``ik`` sums both arm solves;
``pace`` measures rate waiting; ``send`` includes runtime checks and SDK dispatch;
``state`` measures observation joint reads; ``cameras`` includes frame waiting
and image processing; ``outside`` measures time between wrapper ticks, including
outer collection work and logging. Values are per-window maxima in milliseconds,
may come from different ticks and must not be added together. ``rate`` reports
tick frequency and ``fault_ticks`` counts faulted ticks. Timing can reveal camera
waits shorter than the frame timeout and does not modify control targets.

Independent subscribers can consume different frames, and freshness is based
on local reception time. Repeated frozen tracking data is not detected by that
timeout. The model does not supply dual-arm or scene collision avoidance.
This first version is for manual collection, not policy/VR DAgger handoff.
Offline tests validate implementation contracts; physical calibration and
supervised hardware acceptance remain station-specific work.

Overview
--------

PICO VR collection is also available via
``realworld_dual_yam_collect_data_pico``; see :ref:`yam-pico-collection` above.
That mode opens only the two follower CAN chains and three cameras.

The current example covers native environment wiring and demonstration
collection. YAM-specific policy transforms, SFT, and deployment configs are a
separate integration stage.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      No policy required for collection

   .. grid-item-card:: Algorithms
      :text-align: center

      Leader-follower teleoperation · DAgger-ready intervention API

   .. grid-item-card:: Tasks
      :text-align: center

      Dual-arm joint-space demonstrations

   .. grid-item-card:: Hardware
      :text-align: center

      2 followers · 2 motorized leaders · 3 RealSense cameras

| **You'll do:** install the runtime → configure one YAM station → export calibration values → collect successful episodes → inspect both RLinf and LeRobot outputs.
| **Prerequisites:** :doc:`Installation <../../start/installation>` · Linux SocketCAN · three Intel RealSense cameras · a validated ``i2rt`` build · a physical emergency stop.

How RLinf Organizes the Station
-------------------------------

RLinf keeps resource scheduling separate from device control:

.. code-block:: text

   Hydra YAML
     -> NodeHardwareConfig(type="DualYam")
     -> DualYamConfig
     -> DualYamDiscovery.enumerate()          # configuration only
     -> one RobotInfo per complete station
     -> component_placement selects hardware rank 0
     -> WorkerInfo.hardware_infos
     -> RealWorldEnv._create_env()
     -> create_DualYamJointEnv()
     -> optional DualYamLeaderIntervention
     -> optional YamPico device + YamPicoEpisode    # VR collection
     -> DualYamJointEnv
     -> YamControlRuntime                    # only follower-command writer
     -> lazy i2rt backend

One ``DualYamConfig`` describes one complete station: the left/right followers,
left/right leaders, and every camera. It becomes one scheduler hardware rank.
Therefore ``placement: 0`` means "the first complete YAM station"; it does not
mean node rank 0 or CAN interface ``can0``. See
:doc:`Placement <../../concepts/placement>` for the distinction between node and
hardware ranks.

The scheduler only validates configuration and resource ownership. It does not
import ``i2rt``, probe a CAN bus, or open a camera. The assigned environment
worker receives the station as ``RobotInfo`` through
``WorkerInfo.hardware_infos``, and the task factory reads ``robot_info.config``
to reach the same ``DualYamConfig``. Real hardware is opened later, during the
first ``reset()``:

.. code-block:: text

   open all cameras -> warm every camera -> connect left follower -> hold
                    -> connect right follower -> hold -> validate feedback
                    -> connect leaders only when intervention is enabled

Policy-only use leaves leader intervention disabled and never opens either
leader CAN interface. The collection config enables the leader wrapper. All
four CAN interfaces and all cameras must currently be accessible from the same
node as the environment worker.

Configuration Boundaries
~~~~~~~~~~~~~~~~~~~~~~~~

Keep three kinds of settings in their respective YAML sections:

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Configuration
     - Location
     - Owns
   * - Physical station
     - ``cluster.node_groups[].hardware``
     - Node ownership, CAN names, arm/gripper variants, per-device mass and compensation, raw gripper stops, and camera serials.
   * - Task/runtime
     - ``env.eval.override_cfg``
     - Task text, control frequency, RLinf joint/slew limits, timeouts, image size, and leader episode behavior.
   * - Process placement
     - ``cluster.component_placement``
     - Which worker exclusively receives a complete station resource.

Installation
------------

Create the YAM embodied environment using RLinf's standard installer:

.. code-block:: bash

   bash requirements/install.sh embodied --env yam
   source .venv/bin/activate

The installer adds RLinf's embodied dependencies, RealSense/OpenCV, LeRobot,
and the pinned official ``i2rt`` SDK declared in
``requirements/embodied/envs/yam.txt``. It does not require or clone a YAM
application repository, and there is no separate wheel variable or SDK install
path. Modules still import ``i2rt`` lazily, so scheduler-only and dummy imports
remain hardware-free.

The compatible build must support the options used by
``get_yam_robot()`` and must shut down its CAN/control threads reliably. The
adapter deliberately refuses unsupported per-device numeric damping or friction
overrides instead of mutating private SDK arrays.

Before launching RLinf, configure persistent SocketCAN names for the four USB-CAN
adapters and bring those interfaces up using your machine's normal system setup.
The example defaults are:

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Role
     - Default interface
     - Optional override
   * - Left follower
     - ``can_left``
     - ``YAM_LEFT_FOLLOWER_CAN``
   * - Right follower
     - ``can_right``
     - ``YAM_RIGHT_FOLLOWER_CAN``
   * - Left leader
     - ``can_lead_l``
     - ``YAM_LEFT_LEADER_CAN``
   * - Right leader
     - ``can_lead_r``
     - ``YAM_RIGHT_LEADER_CAN``

You can inspect an interface without commanding a robot:

.. code-block:: bash

   ip -details link show can_left
   ip -details link show can_right
   ip -details link show can_lead_l
   ip -details link show can_lead_r

Configure the Station
---------------------

The reusable environment defaults are in
``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``. The complete
single-node collection example is
``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``.

The collection example uses official i2rt auto-calibration by setting each
follower's ``gripper_limits`` to ``null``. On every startup, each follower
moves its gripper in both directions to detect the current ``[closed, open]``
motor range. Keep both grippers completely unobstructed until calibration
finishes.

For a station that deliberately skips startup calibration, replace ``null``
with measured per-station values ordered as ``[closed, open]``:

.. code-block:: bash

   # Raw follower motor positions, ordered [closed, open].
   # The values may be ascending or descending; do not sort them.
   export YAM_LEFT_GRIPPER_CLOSED_RAD=<measured-left-closed>
   export YAM_LEFT_GRIPPER_OPEN_RAD=<measured-left-open>
   export YAM_RIGHT_GRIPPER_CLOSED_RAD=<measured-right-closed>
   export YAM_RIGHT_GRIPPER_OPEN_RAD=<measured-right-open>

   # RealSense serial numbers.
   export YAM_TOP_CAMERA_SERIAL=<top-serial>
   export YAM_LEFT_CAMERA_SERIAL=<left-serial>
   export YAM_RIGHT_CAMERA_SERIAL=<right-serial>

Do not estimate fixed gripper stops. They are raw motor-radian endpoints, not
the normalized ``[0, 1]`` action. Their order carries the motor direction, so a
valid ``[closed, open]`` pair can be decreasing. Fixed multi-turn limits must
also use the encoder revolution active at startup; prefer auto-calibration
unless the installed i2rt build aligns persisted limits to that revolution.

If your CAN aliases differ from the defaults, export the four optional variables
from the table above. All four resolved names must be unique. Camera names and
serials must also be unique.

Per-device tuning is kept beside each device in the hardware configuration:

.. code-block:: yaml

   left_leader:
     ee_mass: null                 # use the pinned i2rt model
     gravity_comp_factor: null     # use the selected arm model's defaults
     grav_comp_kd: null
     coulomb_friction: null
     use_coulomb_friction: false
     bilateral_kp: 0.0
     gripper_invert: false

``ee_mass`` and ``gravity_comp_factor`` affect gravity support;
``grav_comp_kd`` is gravity-compensation damping; ``coulomb_friction`` and
``use_coulomb_friction`` control Coulomb-friction compensation.

The VR recipe explicitly sets both followers' ``gravity_comp_factor`` to
``[1.3, 1.3, 1.3, 1.3, 1.0, 1.0]`` and enables friction compensation.
The local station tuning on 2026-09-08 also sets ``coulomb_friction`` to
``[0.3, 0.3, 0.3, 0.06, 0.06, 0.06]`` in the installed i2rt
``robots/config/yam_v1.yml``. This local SDK edit affects programs using that
model's defaults and is not part of RLinf's dependency pin; recheck it after
reinstalling i2rt. Restart the controller to reload changed settings.
``bilateral_kp`` instead controls leader feedback toward the measured follower
pose and is not a gravity-compensation setting. Keep it at ``0.0`` for initial
collection. Leaving an optional value at ``null`` preserves the pinned SDK
model. Numeric ``grav_comp_kd`` and ``coulomb_friction`` values require an SDK
constructor that explicitly accepts those fields.

The base config's joint limits are the nominal YAM v1 limits. Replace them with
a narrower, validated workspace envelope when the station or task requires it.
For another YAM arm variant, replace both ``arm_type`` and the RLinf joint limits;
startup rejects configured limits outside the SDK's limits.

Configure a Reset Pose
~~~~~~~~~~~~~~~~~~~~~~

The reusable environment config contains a disabled ``reset`` block. First move
the followers to a collision-free start pose, record their measured 14-D state,
and copy the two seven-value arms into the block. Do not use unverified leader
positions. The last value of each arm is the normalized gripper position
(``0=closed``, ``1=open``):

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

Use ``startup`` for collection: the collector calls ``reset()`` after every
clip, but the followers move only once when the process starts. Use ``episode``
for policy deployment to restore the same pose before every episode. ``manual``
moves only when calling ``reset(options={"reset_qpos": True})`` or
``env.unwrapped.reset_to_configured_qpos()`` on the base environment.

Reset motion uses smooth joint-space interpolation at ``step_frequency``,
checks every target against the hard joint limits, holds on failure, and verifies
the measured final error. A straight joint-space path is not collision-aware;
enable the feature only after validating the complete path from every permitted
starting region. When a reset must move followers, active leader synchronization
is released first and must be engaged again after the operator aligns the leaders.

Validate Without Hardware
-------------------------

Run the YAM unit tests before a supervised hardware session. They use mocks and
must not open CAN or cameras:

.. code-block:: bash

   pytest -q \
     tests/unit_tests/test_yam_hardware.py \
     tests/unit_tests/test_yam_runtime.py \
     tests/unit_tests/test_yam_env.py \
     tests/unit_tests/test_yam_intervention.py \
     tests/unit_tests/test_yam_imports.py \
     tests/unit_tests/test_yam_examples.py

This checks registration, configuration conversion, 14-D ordering, joint and
slew limiting, stale/non-finite handling, intervention ownership, cleanup, and
lazy ``i2rt`` imports. It is not a substitute for low-speed hardware acceptance.

Collect Demonstrations
----------------------

.. danger::

   The next command is the hardware launch point. On its first ``reset()``, it
   opens cameras, connects both followers, and then connects both leaders.
   Do not run it as a configuration-only check.

From the RLinf repository root, start a 50-episode collection:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
     realworld_dual_yam_collect_data

The supplied config collects 50 episodes and stores one tabletop-tidying
instruction as the task string. To keep the launcher in RLinf's existing
config-name style, copy or edit the following fields in
``realworld_dual_yam_collect_data.yaml`` when creating another recipe:

.. code-block:: yaml

   runner:
     num_data_episodes: 50
   env:
     eval:
       override_cfg:
         task_description: "Tidy up the table. ..."   # the shipped instruction

What this does:

1. launches RLinf's generic ``collect_real_data.py`` entry point;
2. asks the scheduler for one complete ``DualYam`` resource;
3. constructs ``RealWorldEnv`` and the registered ``DualYamJointEnv-v1`` task;
4. enables motorized-leader intervention and button-controlled episodes;
5. writes recorded episodes directly to LeRobot, skipping the RLinf replay buffer.

There is no ``--convert`` step and no runtime clone/import of a YAM application
repository.

Teaching-Handle Controls
~~~~~~~~~~~~~~~~~~~~~~~~

Release both buttons while leaders connect. The first sample establishes the
idle electrical level for each handle. A button on either handle controls the
whole dual-arm station.

.. list-table::
   :header-rows: 1
   :widths: 25 32 43

   * - Control
     - State
     - Result
   * - Top / first button
     - Synchronization off
     - Smoothly engage from the measured follower pose to the current leader pose, then let both leaders command both followers.
   * - Top / first button
     - Synchronization on
     - Hold both followers and return both leaders to gravity-compensation idle.
   * - Record / second button
     - Waiting before an episode
     - Start a new recorded episode at the current pose.
   * - Record / second button
     - Recording
     - End the episode as a success with reward ``1`` while keeping teleoperation synchronized for the next episode.
   * - Teaching trigger
     - Default mapping
     - Released is gripper ``1`` (open); pressed is ``0`` (closed). Set ``gripper_invert: true`` per leader to reverse it.

The example uses ``sync_on_reset: false``: press the top button once when you
are ready to take control. ``preserve_sync_between_episodes: true`` makes the
record button split episodes without releasing that control; only the top
button toggles synchronization. It also uses ``unsynced_action_source: hold``,
so the collector's placeholder zero action can never send the followers toward
zero. Button events are rising-edge-triggered and debounced.

The terminal reports waiting, recording, and stopped states only on transitions.
The separate background-save message confirms that the named episode is on disk;
it does not change the current recording state. The progress bar is only a count.

Teaching-handle collection enables ``data_collection.export_mp4: true``. After
each episode is saved, a separate low-priority process exports per-view H.264
MP4s serially without blocking recording. Videos are written under
``collected_data/rank_0/review_videos/id_N/episode_XXXXXX/`` as ``top.mp4``,
``left.mp4``, and ``right.mp4``. Save completion and video start/completion messages
are suppressed; export failures are still reported. New recordings may continue
while video jobs queue. Normal shutdown
releases the robot interface before waiting for remaining videos. These are
review artifacts; training continues to use the original Parquet data.

Observation and Action Contract
-------------------------------

Every state and action uses the same absolute 14-D layout:

.. code-block:: text

   [left_q0, ..., left_q5, left_gripper,
    right_q0, ..., right_q5, right_gripper]

Arm joints are radians. Grippers are normalized to ``0=closed, 1=open``.
Observations contain measured follower positions. Commands are always checked
for the exact shape and finite values, and grippers are clipped to their valid
range. By default, arm commands are also clipped to the configured limits and
limited by ``max_joint_delta`` relative to the measured pose. The collection
example sets ``enforce_runtime_joint_limits: false`` to match the legacy
yam-abc teleoperation path: after smooth engagement, leader joints pass directly
to i2rt, which applies its hardware limits. The wrapper reports the resulting
target as ``intervene_action``.

Camera and dataset names change at three deliberate boundaries:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Boundary
     - Keys
     - Meaning
   * - YAM Gym observation
     - ``frames.top_rgb``, ``frames.left_rgb``, ``frames.right_rgb``
     - Named RGB frames; ``state.joint_position`` is the 14-D measured state.
   * - ``RealWorldEnv``
     - ``main_images``, ``extra_view_images``
     - ``top_rgb`` becomes the main image. Remaining names are sorted, so index 0 is ``left_rgb`` and index 1 is ``right_rgb`` for the supplied config.
   * - RLinf LeRobot writer
     - ``image``, ``extra_view_image-0``, ``extra_view_image-1``
     - Literal dataset feature names for top, left, and right respectively. State/action features are ``state`` and ``actions``.

Keep the supplied camera names if downstream transforms depend on that ordering.
The current generic collector preserves view order but not the semantic
``left_rgb``/``right_rgb`` names in the final LeRobot columns. A future YAM
policy dataconfig must map the literal dataset keys explicitly.

Output Layout
-------------

``collect_data.sh`` creates a fresh ``logs/<timestamp>/`` directory. The
successful episode is written to LeRobot under it:

.. code-block:: text

   logs/<timestamp>/
   `-- collected_data/
       `-- rank_0/
           `-- id_0/                 # LeRobot shard for this run
               |-- meta/info.json
               |-- meta/episodes.jsonl
               |-- meta/tasks.jsonl
               |-- meta/stats.json
               |-- data/...
               |-- videos/...        # layout depends on the installed LeRobot version
               |-- recording_errors.jsonl          # only after an overflow
               `-- invalid_episodes/overflow_XXXX/ # isolated prefix for one overflow
                   |-- frames.pkl
                   `-- <camera keys>/

The example enables ``streaming: true`` and disables ``runner.save_demos``:
every recorded frame is written to the LeRobot dataset as it is captured
by the existing export thread. LeRobot v2 uses lossless, uncompressed PNG to
reduce CPU overhead, without a second image queue. At 240 pending tasks, the
current recording is stopped with ``recording_invalid`` instead of blocking
teleoperation. Its contiguous prefix is moved into a fresh
``invalid_episodes/overflow_XXXX/`` directory as images and ``frames.pkl``, and
the reason is appended to ``recording_errors.jsonl`` at the shard root; the
prefix is excluded from normal LeRobot metadata and success counts.
End that recording and restart after the writer catches up. Uncompressed PNG
requires more temporary disk space and write bandwidth.
For LeRobot v2, episode saving embeds images in batches of 16
frames and publishes the Parquet file after writing completes. Saved episodes
use separate ``id_N`` shards in streaming mode. A separate worker saves and
finalizes completed writers serially while new frames go to the next shard.
MP4 export reads batches within one Parquet row group at a time to avoid
retaining image buffers across row groups. Saved episodes
are never concatenated into an in-memory ``hf_dataset``. The current episode's
states, actions, image paths, and indices still occupy memory, so the whole
collector does not have strictly constant memory usage. Successful and failed episodes are
both saved; the episode-level ``is_success`` flag is stamped when the episode
ends, so training can filter on it. The ``demos/`` directory is no longer
produced. ``finalize_interval`` applies only to non-streaming mode; streaming
writers are released after each episode is saved. LeRobot v2 episode metadata and Parquet are saved
when ``save_episode`` returns successfully; pressing the success button only
submits that work and does not confirm persistence.
Set ``resume: true`` only when reusing an explicit ``save_dir``;
a resumed run writes a new ``id_N`` shard and does not overwrite finalized
shards.

The LeRobot frames include ``state``, ``actions``, ``image``,
``extra_view_image-0``, ``extra_view_image-1``, ``done``, ``is_success``,
``intervene_flag``, and ``segment_id``. The task string is stored through
LeRobot's task metadata. See :doc:`Data Collection <../../guides/data_collection>`
for the generic writer behavior.

Implementation Map
------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - File
     - Responsibility
   * - ``rlinf/robotics/robots/dual_yam.py``
     - Defines one complete station resource, converts nested Hydra config, validates CAN/camera uniqueness, and registers the robot type without hardware I/O.
   * - ``rlinf/envs/real/yam/types.py``
     - Defines the 14-D state/action contract, typed state objects, command results, and backend protocols.
   * - ``rlinf/envs/real/yam/config.py``
     - Validates task-level timing, limits, camera timeouts, and leader-intervention behavior.
   * - ``rlinf/envs/real/yam/i2rt_backend.py``
     - Isolates the lazy ``i2rt`` import and adapts follower, leader, handle, health, and cleanup APIs.
   * - ``rlinf/envs/real/yam/mock_backend.py``
     - Supplies hardware-free follower/leader implementations for dummy use and tests.
   * - ``rlinf/envs/real/yam/control_runtime.py``
     - Owns all transports, serializes follower writes, validates every command, engages leaders smoothly, holds on failures, and closes in safe order.
   * - ``rlinf/envs/real/yam/dual_yam_joint_env.py``
     - Implements the Gym action/observation spaces, lazy startup, camera processing, step pacing, and resource cleanup.
   * - ``rlinf/envs/real/yam/leader_intervention.py``
     - Implements dual-leader synchronization, buttons, episode control, policy/hold ownership, and ``intervene_action`` reporting.
   * - ``rlinf/envs/real/yam/__init__.py``
     - Exposes the public YAM API and registers ``DualYamJointEnv-v1`` with the task registry.
   * - ``rlinf/envs/real/yam/pico_episode.py``
     - Record, discard, and keyboard episode control for VR collection: ``YamPicoEpisode``.
   * - ``rlinf/robotics/parts/teleop/yam_pico.py``
     - The ``yam_pico`` teleop device: PICO readings mapped to the 14-value joint target, IK, and fault holds.
   * - ``examples/embodiment/config/env/realworld_dual_yam_joint.yaml``
     - Reusable real-world Gym/task defaults and explicit RLinf safety limits.
   * - ``examples/embodiment/config/realworld_dual_yam_collect_data.yaml``
     - Complete one-station scheduler, teleoperation, and direct LeRobot collection recipe.
   * - ``requirements/install.sh`` (``--env yam``)
     - Builds one complete YAM environment, including camera, LeRobot, and the pinned i2rt SDK, without requiring a YAM application repository.
   * - ``requirements/embodied/envs/yam.txt``
     - Pins the official i2rt commit and declares the native runtime's camera and configuration dependencies.
   * - ``requirements/embodied/envs/yam-build-constraints.txt``
     - Keeps i2rt's ruckig source-build constraint local to the YAM environment.
   * - ``examples/embodiment/collect_data.sh``
     - Generic collection launcher shared by every real-world recipe; it selects the config by name and creates the timestamped log directory. The YAM change adds the explicit entry-point and log-file variables while leaving the default log path generic.
   * - ``rlinf/envs/real/__init__.py``
     - Imports the YAM task package so Gym registration is available through RLinf's real-world environment entry point.
   * - ``rlinf/robotics/robots/__init__.py``, ``rlinf/robotics/__init__.py``, and ``rlinf/envs/real/__init__.py``
     - Load the YAM robot module so ``DualYam`` reaches the hardware-policy registry, and import the task package so a cluster config naming ``DualYam`` resolves.
   * - ``tests/unit_tests/test_yam_hardware.py``
     - Covers registry conversion, station enumeration, ownership conflicts, and direction-preserving gripper calibration.
   * - ``tests/unit_tests/test_yam_runtime.py``
     - Covers lazy connection, command safety, stale feedback, role-specific i2rt modes, and cleanup.
   * - ``tests/unit_tests/test_yam_env.py``
     - Covers the dummy Gym observation/action contract, 14-D ordering, and idempotent close.
   * - ``tests/unit_tests/test_yam_intervention.py``
     - Covers policy/leader ownership, synchronization failure cleanup, and episode termination behavior.
   * - ``tests/unit_tests/test_yam_imports.py``
     - Guards the hardware-free import path and verifies that ``i2rt`` remains lazy.
   * - ``tests/unit_tests/test_yam_examples.py``
     - Guards the public YAML contract, direct LeRobot settings, and bundled pinned-SDK/no-application-repository installation rule.

Known Limits
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Area
     - Current boundary
   * - Control representation
     - Absolute 14-D joint space only. TCP/Cartesian actions require a separately versioned environment and dataset contract.
   * - Cameras
     - RealSense RGB only; ``enable_depth`` must remain ``false``.
   * - Placement
     - Followers, motorized leaders, and cameras must be local to one environment worker node. Remote per-arm controllers are not implemented.
   * - Intervention
     - Synchronization and intervention apply to both arms together. There is no independent left/right intervention mask yet.
   * - Models
     - This example collects data. A YAM-specific model dataconfig, normalization statistics, SFT recipe, and policy deployment example are not included yet.
   * - SDK tuning
     - Gravity factor and friction enablement use public SDK options. Numeric per-device damping/friction overrides require a compatible ``i2rt`` constructor and are rejected otherwise.
   * - Failure response
     - Software performs best-effort measured-pose hold and cleanup; it is not a physical emergency stop.
   * - Validation
     - Unit tests use mocks. Validate the pinned SDK and station at low speed, one arm at a time, before bimanual collection.

Foot switch confirmation
~~~~~~~~~~~~~~~~~~~~~~~~

Set ``leader_intervention.foot_switch_device`` to the pedal's stable
``/dev/input/by-id/`` path; ``null`` restores immediate saving at the handle
record boundary. The tested PCsensor maps left to ``KEY_A`` (30, discard) and
right to ``KEY_C`` (46, keep). On this station the middle pedal is ``KEY_B``
(48), enabled for return to the operator-confirmed home. The handle record
button starts/stops recording, and its sync button controls teleop.

After the operator confirms the synchronized teaching-arm home, set
``reset.enabled: true``, ``reset.mode: manual`` and its ``left_qpos/right_qpos``.
Set ``leader_intervention.foot_switch_reset_key`` to the measured middle key code.
The middle pedal then returns both leaders and both followers to that absolute
joint target, one frame per environment step. A return during recording remains
in the same episode; press the white button after completion to end recording.
Grippers remain controlled by the passive teaching-handle triggers. The yellow
button cancels the return, holds followers and restores leader gravity idle.
Completion also restores the original leader gains. This pose is not an encoder
zero and does not trigger automatic startup or episode-boundary movement.
Validate the first return slowly with empty grippers and a clear path: joint
interpolation does not perform collision planning. Review ignores the middle pedal.


After recording stops, progress includes one provisional episode while awaiting
a pedal decision. No review-time frames are recorded; teleoperation remains
available. Keep publishes the episode and queues three MP4s. Discard removes
only the pending episode's temporary images and state sidecar entries, and
reverses its provisional progress. Earlier kept episodes remain untouched.
Press the handle record button to start again. Autorepeat and stale presses do
not make decisions. Normal close or early reset discards an unconfirmed review;
even the last episode requires confirmation before collection completes.

The input requires evdev permissions. On YAM Box, the operator belongs to
``plugdev`` and ``/etc/udev/rules.d/70-yam-foot-switch.rules`` contains:

.. code-block:: text

   SUBSYSTEM=="input", KERNEL=="event*", ATTRS{idVendor}=="3553", ATTRS{idProduct}=="b001", GROUP="plugdev", MODE="0660"

Reload udev rules and reconnect the pedal after installation. Only the configured
pedal is exclusively grabbed; the operator's regular keyboard is untouched.
