

import collections
import dataclasses
import cv2
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH   = os.path.join(SCRIPT_DIR, "conveyor.xml")

# ── Belt / spawn constants ─────────────────────────────────────────────────
N_OBJECTS      = 10
BELT_VELOCITY  = 0.200        # m/s  (+X)
SPAWN_INTERVAL = 1.5          # s
SPAWN_X        = 0.10         # m
SPAWN_Y        = 0.00         # m
SPAWN_Z        = 0.556        # m   belt top (0.526) + half-height (0.030)
SPAWN_QUAT     = [1, 0, 0, 0]

# Belt-zone bounds for kinematic override
BELT_X_MIN  =  0.00
BELT_X_MAX  =  2.00
BELT_Y_MAX  =  0.12
BELT_Z_MIN  =  0.540
BELT_Z_MAX  =  0.640

# ── Settle detection ───────────────────────────────────────────────────────
# Fires when object is off-belt (end-of-belt OR pushed sideways) AND at rest.
SETTLE_Z               = 0.25   # m
SETTLE_SPEED           = 0.08   # m/s
SETTLE_MIN_ACTIVE_TIME = 3.0    # s  (prevents false trigger right after spawn)
SETTLE_Y_PUSH_THRESH   = 0.28   # |y| > this → object was pushed off belt side

# ── Vision ─────────────────────────────────────────────────────────────────
CAMERA_NAME        = "side_camera"
CAMERA_X           = 0.40    # world-X of the camera; detection window centred here
CAMERA_DETECT_WIN  = 0.07    # m  — ±window around CAMERA_X to trigger a render
VISION_EVERY_STEPS = 5       # render every N sim steps  (= 0.010 s at dt=0.002)
RENDER_W           = 320
RENDER_H           = 240

# HSV colour ranges (OpenCV H: 0–180, S/V: 0–255)
#   Pure-red MuJoCo material (1,0,0) → H≈0, S=255, V=255
#   Pure-yellow (1,1,0)             → H≈30, S=255, V=255
HSV_RED_LO1    = np.array([  0, 120,  80], np.uint8)
HSV_RED_HI1    = np.array([ 10, 255, 255], np.uint8)
HSV_RED_LO2    = np.array([168, 120,  80], np.uint8)   # hue wraps near 180
HSV_RED_HI2    = np.array([180, 255, 255], np.uint8)
HSV_YELLOW_LO  = np.array([ 22, 120,  80], np.uint8)
HSV_YELLOW_HI  = np.array([ 38, 255, 255], np.uint8)
COLOR_MIN_PX   = 60   # minimum pixels in ROI to confirm a detection

# ── Pusher constants ───────────────────────────────────────────────────────
#
#  Geometry tính toán:
#    Pusher body world Y = 0.20 + q   (q âm khi extend)
#    Face front Y        = 0.20 + q - 0.010   (half-thickness mặt = 0.010)
#    Object center Y     = face_front + 0.030  (half-size vật)
#
#    Để object center qua rìa belt (Y = -0.125) cần margin:
#      Object center target = -0.22
#      Face front target    = -0.22 - 0.030 = -0.25
#      q = face_front - 0.20 + 0.010 = -0.25 - 0.20 + 0.010 = -0.44
#    → dùng PUSHER_EXTENDED = -0.42  (face → -0.23, object center → -0.20)
#
PUSHER_RED_X      = 0.80
PUSHER_YELLOW_X   = 1.480  # body pos của pusher yellow trong XML

# Ground-truth detection point cho YELLOW — detect gần pusher thay vì từ camera X=0.40
# Delay chỉ (1.480−1.35)/0.200 = 0.65 s thay vì 5.4 s → ít sai lệch hơn nhiều
YELLOW_DETECT_X   = 1.35
YELLOW_OBJECTS    = frozenset([2, 5, 8])   # index vật màu vàng (cố định theo material)

# Stroke geometry (body Y=0.20, face half-thick=0.010, object half-size=0.030):
#   face_front = 0.20 + q - 0.010
#   object_center_Y = face_front + 0.030 = 0.22 + q
#
#   RED    q=-0.38 → face Y≈-0.19, object center ≈ -0.16  (đủ momentum vào red bin Y=-0.40)
#   YELLOW q=-0.42 → face Y≈-0.23, object center ≈ -0.20  (đủ momentum vào yellow bin Y=-0.40)
PUSHER_RED_EXTENDED    = -0.24 # tăng từ -0.20: đẩy mạnh hơn, object bay vào red bin
PUSHER_YELLOW_EXTENDED = -0.24 # tăng từ -0.30: đẩy đủ xa để object rơi vào yellow bin
PUSHER_RETRACTED  =  0.00
PUSH_HOLD_TIME    =  0.8   # s — đủ để vật clear belt, pusher retract kịp trước object tiếp theo

# Retract khi object đã qua rìa belt đủ xa để có momentum vào bin
# RED -0.38 → object center -0.16 ; YELLOW -0.42 → object center -0.20
# Dùng threshold 0.22 để chắc chắn object đã có vận tốc đủ trước khi retract
PUSH_CLEAR_Y      =  0.22  # m  (tăng từ 0.13 → giữ pusher lâu hơn, tích momentum tốt hơn)

# ── Initial staging positions (Phase-1 only) ──────────────────────────────
_Z_IDLE = 0.050
IDLE_POS: dict[int, list[float]] = {
    0: [0.7625, -0.4375, _Z_IDLE],  # Red bin 2×2
    3: [0.8375, -0.4375, _Z_IDLE],
    6: [0.7625, -0.3625, _Z_IDLE],
    9: [0.8375, -0.3625, _Z_IDLE],
    1: [2.25,  -0.075,   _Z_IDLE],  # Blue bin row
    4: [2.25,   0.000,   _Z_IDLE],
    7: [2.25,   0.075,   _Z_IDLE],
    2: [1.425, -0.400,   _Z_IDLE],  # Yellow bin row
    5: [1.500, -0.400,   _Z_IDLE],
    8: [1.575, -0.400,   _Z_IDLE],
}


# ═══════════════════════════════════════════════════════════════════════════
#  detect_color
# ═══════════════════════════════════════════════════════════════════════════

def detect_color(frame_bgr: np.ndarray) -> str:
    """
    Identify the dominant conveyor-object colour in a BGR camera frame.

    Strategy:
      • Use the middle 60 % (both axes) of the frame as the ROI.
        The side_camera is angled so objects at CAMERA_X appear in
        the upper-centre region; searching 20–80 % of height catches them.
      • Convert ROI to HSV and apply separate masks for red (two hue
        ranges because red wraps near H=0/180) and yellow.
      • Return 'R', 'Y', or 'NONE'.  Blue objects return 'NONE' — no
        push is scheduled and they fall off the belt end naturally.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR image from cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR).

    Returns
    -------
    'R'    – red object detected in ROI
    'Y'    – yellow object detected in ROI
    'NONE' – no sortable colour found (blue object or empty belt)
    """
    h, w = frame_bgr.shape[:2]

    # ROI: centre 60 % of frame in both dimensions
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    roi = frame_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        return 'NONE'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Red mask (two ranges — hue wraps around 0°/180°) ──────────────────
    mask_r1  = cv2.inRange(hsv, HSV_RED_LO1, HSV_RED_HI1)
    mask_r2  = cv2.inRange(hsv, HSV_RED_LO2, HSV_RED_HI2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # ── Yellow mask ────────────────────────────────────────────────────────
    mask_yellow = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)

    n_red    = int(cv2.countNonZero(mask_red))
    n_yellow = int(cv2.countNonZero(mask_yellow))

    if n_red >= COLOR_MIN_PX and n_red >= n_yellow:
        return 'R'
    if n_yellow >= COLOR_MIN_PX:
        return 'Y'
    return 'NONE'


# ═══════════════════════════════════════════════════════════════════════════
#  PushScheduler
# ═══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class _PushEvent:
    fire_time:   float   # sim_time để extend pusher
    max_retract: float   # safety fallback — retract dù vật chưa clear
    act_id:      int     # data.ctrl index
    obj_idx:     int     # object đang bị đẩy (exempt khỏi kinematic override)
    qpos_adr:    int     # địa chỉ qpos của vật để monitor Y
    color:       str     # 'R' hoặc 'Y' — để chọn đúng stroke khi fire
    fired:       bool = False
    retracted:   bool = False


class PushScheduler:
    """
    Translates vision detections into timed pusher actuations.

    Cải tiến so với bản gốc:
      - Retract thông minh: chờ vật thực sự ra khỏi belt (|Y| > PUSH_CLEAR_Y)
        thay vì dùng timer cố định → đảm bảo vật không bị kéo ngược lại.
      - Safety fallback: nếu sau PUSH_HOLD_TIME vật vẫn chưa clear, retract luôn
        để pusher không kẹt vĩnh viễn.

    Usage:
        schedule(color, sim_time, obj_idx)  — called on each detection
        step(sim_time)                      — called every sim step
        active_pushes                       — set[int] of object indices
                                              currently being pushed
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 manager: "ObjectQueueManager") -> None:
        self.model   = model
        self.data    = data
        self._mgr    = manager   # cần để đọc spawn_time khi guard yellow re-push

        # Cache actuator ctrl indices by name
        def _act(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{name}' not found in model.")
            return aid

        self._act_id: dict[str, int] = {
            'R': _act("act_push_red"),
            'Y': _act("act_push_yellow"),
        }

        # Cache qpos addresses của tất cả objects để monitor Y khi đẩy
        self._obj_qpos_adr: list[int] = []
        for i in range(N_OBJECTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            if jid < 0:
                raise RuntimeError(f"Joint 'obj{i}_free' not found in model.")
            self._obj_qpos_adr.append(model.jnt_qposadr[jid])

        self._pending:   list[_PushEvent]   = []

        # Objects currently under active push — exempt from belt kinematic vy=0
        self.active_pushes: set[int] = set()

        # BUG-FIX (yellow ghost push): sim_time lúc event FIRE cho từng obj_idx.
        # Dùng để chặn re-schedule vật vừa bị đẩy cho đến khi nó được recycle
        # (manager.spawn_time[i] > _last_fire_sim_time[i]).
        self._last_fire_sim_time: dict[int, float] = {}

        # Ensure both pushers start retracted
        for aid in self._act_id.values():
            data.ctrl[aid] = PUSHER_RETRACTED

    def schedule(self, color: str, det_time: float, obj_idx: int,
                 obj_x: float) -> None:
        """
        Schedule a push event cho `color` được detect tại `det_time`.

        Fixes (v2):
          1. Delay tính từ X thực tế của vật lúc detect → pusher fire đúng lúc.
          2. BUG-FIX (yellow ghost): chặn re-schedule nếu vật này đã được fired
             trong chu kỳ belt hiện tại (chưa được recycle).
          3. BUG-FIX (red miss): thay vì chặn hoàn toàn khi cùng act_id pending,
             chỉ chặn khi fire_time MỚI bị overlap với max_retract của event CŨ.
             → 2 vật đỏ liên tiếp có fire_time không overlap → cả 2 đều được schedule.
        """
        if color not in ('R', 'Y'):
            return   # Blue — không đẩy

        act_id = self._act_id[color]

        # ── Guard 1: không schedule lại cùng một obj_idx đang có event pending ──
        if any(ev.obj_idx == obj_idx for ev in self._pending):
            return

        # ── Guard 2 (yellow ghost-push fix): nếu obj này đã được fired trước đó,
        #    chỉ cho schedule lại khi nó đã được recycle (spawn_time mới hơn) ──────
        if obj_idx in self._last_fire_sim_time:
            if self._mgr.spawn_time[obj_idx] <= self._last_fire_sim_time[obj_idx]:
                return   # Chưa recycle — chặn

        # Tính delay chính xác từ vị trí X hiện tại của vật
        pusher_x  = PUSHER_RED_X if color == 'R' else PUSHER_YELLOW_X
        delay     = max(0.05, (pusher_x - obj_x) / BELT_VELOCITY)
        fire_time = det_time + delay
        max_ret   = fire_time + PUSH_HOLD_TIME   # safety fallback

        # ── Guard 3 (red-miss fix): chỉ chặn nếu fire_time MỚI overlap với
        #    max_retract của event cũ (cùng actuator) — không chặn nếu không overlap ─
        for ev in self._pending:
            if ev.act_id == act_id and fire_time <= ev.max_retract:
                return   # Xung đột thực sự — pusher còn đang extend

        self._pending.append(_PushEvent(
            fire_time   = fire_time,
            max_retract = max_ret,
            act_id      = act_id,
            obj_idx     = obj_idx,
            qpos_adr    = self._obj_qpos_adr[obj_idx],
            color       = color,
        ))

    def _obj_clear_of_belt(self, ev: _PushEvent) -> bool:
        """
        Trả về True khi vật đã ra khỏi belt hoàn toàn.
        Điều kiện: |Y| > PUSH_CLEAR_Y (vật đã vượt qua rail -Y = -0.138).
        """
        y = float(self.data.qpos[ev.qpos_adr + 1])
        return abs(y) > PUSH_CLEAR_Y

    def step(self, sim_time: float) -> None:
        """
        Fire / retract pushers theo logic thông minh.
        Gọi mỗi simulation step (sau mj_step).

          Bước 1: Khi đến fire_time  → extend pusher (PUSHER_EXTENDED).
          Bước 2: Sau khi fired, mỗi bước kiểm tra Y của vật:
                    - Phải đợi ít nhất 0.3s sau fire trước khi check clear
                      (để pusher có thời gian extend thực sự).
                    - |Y| > PUSH_CLEAR_Y  → vật đã ra belt → retract ngay.
                    - sim_time > max_retract → safety timeout → retract luôn.
          Bước 3: Sau retract → xoá event khỏi pending list.
        """
        done = []
        for ev in self._pending:

            # ── Bước 1: Fire — mở rộng pusher ────────────────────────────────
            if not ev.fired and sim_time >= ev.fire_time:
                stroke = PUSHER_RED_EXTENDED if ev.color == 'R' else PUSHER_YELLOW_EXTENDED
                self.data.ctrl[ev.act_id] = stroke
                self.active_pushes.add(ev.obj_idx)
                # BUG-FIX (yellow ghost): ghi nhận thời điểm fire để guard re-schedule
                self._last_fire_sim_time[ev.obj_idx] = sim_time
                ev.fired = True

            # ── Bước 2: Kiểm tra điều kiện retract ───────────────────────────
            if ev.fired and not ev.retracted:
                # Chờ ít nhất 0.3s sau fire để pusher extend xong
                min_hold_ok  = sim_time >= ev.fire_time + 0.3
                obj_cleared  = min_hold_ok and self._obj_clear_of_belt(ev)
                time_expired = sim_time >= ev.max_retract

                if obj_cleared or time_expired:
                    self.data.ctrl[ev.act_id] = PUSHER_RETRACTED
                    self.active_pushes.discard(ev.obj_idx)
                    ev.retracted = True

            if ev.retracted:
                done.append(ev)

        for ev in done:
            self._pending.remove(ev)




# ═══════════════════════════════════════════════════════════════════════════
#  ObjectQueueManager
# ═══════════════════════════════════════════════════════════════════════════

class ObjectQueueManager:
    """
    FIFO infinite conveyor loop.

    idle_queue   — initial pool (Phase 1). Drained once at startup.
    fallen_queue — FIFO of settled objects waiting to be re-spawned.

    Settle condition (per 'active' object, evaluated in step()):
      off_belt = x > BELT_X_MAX          (fell off belt end)
             OR |y| > SETTLE_Y_PUSH_THRESH  (pushed sideways into bin)
      settled  = off_belt AND z < SETTLE_Z AND |v| < SETTLE_SPEED
               AND active for >= SETTLE_MIN_ACTIVE_TIME

    run_conveyor(excluded):
      Belt kinematic override is skipped for objects in `excluded`
      (objects currently being pushed by a servo pusher).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data  = data

        self.qpos_adr: list[int] = []
        self.qvel_adr: list[int] = []
        for i in range(N_OBJECTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"obj{i}_free")
            if jid < 0:
                raise RuntimeError(f"Joint 'obj{i}_free' not found in model.")
            self.qpos_adr.append(model.jnt_qposadr[jid])
            self.qvel_adr.append(model.jnt_dofadr[jid])

        # States: 'idle' | 'active' | 'fallen'
        self.state:        list[str]              = ['idle'] * N_OBJECTS
        self.idle_queue:   collections.deque[int] = collections.deque(range(N_OBJECTS))
        self.fallen_queue: collections.deque[int] = collections.deque()
        self.last_spawn:   float                  = -SPAWN_INTERVAL
        self.spawn_time:   dict[int, float]       = {i: 0.0 for i in range(N_OBJECTS)}

        # Park all objects at staging positions
        for i in range(N_OBJECTS):
            self._write_pose(i, IDLE_POS[i], SPAWN_QUAT)

    # ── Low-level helpers ──────────────────────────────────────────────────

    def _write_pose(self, idx: int,
                    pos:  list | np.ndarray,
                    quat: list | np.ndarray | None = None) -> None:
        qa = self.qpos_adr[idx]
        self.data.qpos[qa:qa + 3] = pos
        if quat is not None:
            self.data.qpos[qa + 3:qa + 7] = quat
        self.data.qvel[self.qvel_adr[idx]:self.qvel_adr[idx] + 6] = 0.0

    def _read_pos(self, idx: int) -> tuple[float, float, float]:
        qa = self.qpos_adr[idx]
        return (float(self.data.qpos[qa]),
                float(self.data.qpos[qa + 1]),
                float(self.data.qpos[qa + 2]))

    def _read_speed(self, idx: int) -> float:
        va = self.qvel_adr[idx]
        vx = float(self.data.qvel[va])
        vy = float(self.data.qvel[va + 1])
        vz = float(self.data.qvel[va + 2])
        return float(np.sqrt(vx*vx + vy*vy + vz*vz))

    # ── Spawn ──────────────────────────────────────────────────────────────

    def _do_spawn(self, idx: int, sim_time: float) -> None:
        self._write_pose(idx, [SPAWN_X, SPAWN_Y, SPAWN_Z], SPAWN_QUAT)
        va = self.qvel_adr[idx]
        self.data.qvel[va]     = BELT_VELOCITY
        self.data.qvel[va + 1] = 0.0
        self.data.qvel[va + 2] = 0.0
        self.state[idx]       = 'active'
        self.spawn_time[idx]  = sim_time

    # ── Belt kinematic override ────────────────────────────────────────────

    def run_conveyor(self, excluded: set[int] | None = None) -> None:
        """
        Inject vx=BELT_VELOCITY for active objects on the belt surface.

        Parameters
        ----------
        excluded : set[int], optional
            Object indices currently being pushed by a servo pusher.
            These are skipped so the pusher vy impulse is not zeroed.
        """
        excluded = excluded or set()

        for i in range(N_OBJECTS):
            if self.state[i] != 'active':
                continue
            if i in excluded:
                continue   # pusher is in control — do not override

            qa      = self.qpos_adr[i]
            x, y, z = (float(self.data.qpos[qa]),
                       float(self.data.qpos[qa + 1]),
                       float(self.data.qpos[qa + 2]))

            if x > BELT_X_MAX:
                continue   # past belt end — free-fall, hands off

            on_belt = (BELT_X_MIN <= x <= BELT_X_MAX and
                       abs(y)     <= BELT_Y_MAX       and
                       BELT_Z_MIN <= z <= BELT_Z_MAX)
            if not on_belt:
                continue

            va = self.qvel_adr[i]
            self.data.qvel[va]     = BELT_VELOCITY
            self.data.qvel[va + 1] = 0.0
            self.data.qvel[va + 2] = 0.0
            self.data.qvel[va + 3] = 0.0
            self.data.qvel[va + 4] = 0.0
            self.data.qvel[va + 5] = 0.0
            self.data.qpos[qa + 3:qa + 7] = SPAWN_QUAT

    # ── Settle detection + FIFO spawn ─────────────────────────────────────

    def step(self, sim_time: float) -> None:
        """
        1. Detect settled objects (fell off belt end OR pushed sideways).
        2. Drain idle_queue first, then fallen_queue each SPAWN_INTERVAL.
        """

        # ── Settle detection ───────────────────────────────────────────────
        for i in range(N_OBJECTS):
            if self.state[i] != 'active':
                continue
            if (sim_time - self.spawn_time[i]) < SETTLE_MIN_ACTIVE_TIME:
                continue

            x, y, z = self._read_pos(i)
            speed    = self._read_speed(i)

            # Off-belt: either past the passive drum OR pushed sideways into bin
            off_belt = (x > BELT_X_MAX) or (abs(y) > SETTLE_Y_PUSH_THRESH)

            if off_belt and z < SETTLE_Z and speed < SETTLE_SPEED:
                self.state[i] = 'fallen'
                self.fallen_queue.append(i)

        # ── Spawn timer ────────────────────────────────────────────────────
        if (sim_time - self.last_spawn) < SPAWN_INTERVAL:
            return

        if self.idle_queue:
            idx = self.idle_queue.popleft()
            self._do_spawn(idx, sim_time)
            self.last_spawn = sim_time

        elif self.fallen_queue:
            idx = self.fallen_queue.popleft()   # FIFO — first settled → first re-used
            self._do_spawn(idx, sim_time)
            self.last_spawn = sim_time
        # else: no objects available yet — wait


# ═══════════════════════════════════════════════════════════════════════════
#  Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════

def run() -> None:
    """
    Simulation loop.  Step order each iteration:
      1. mj_step            — advance physics
      2. run_conveyor()     — kinematic belt override (skip pushed objects)
      3. push_sched.step()  — fire / retract pushers on schedule
      4. manager.step()     — settle detection + spawn timer
      5. Vision block       — render + detect_color → push_sched.schedule()
      6. viewer.sync()      — update viewer
    """
    if not os.path.isfile(XML_PATH):
        raise FileNotFoundError(
            f"Model file not found: {XML_PATH}\n"
            "Place conveyor.xml in the same directory as this script.")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── Subsystem initialisation ───────────────────────────────────────────
    manager    = ObjectQueueManager(model, data)
    push_sched = PushScheduler(model, data, manager)   # BUG-FIX: manager ref cho yellow ghost guard
    renderer   = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

    if cam_id < 0:
        raise RuntimeError(f"Camera '{CAMERA_NAME}' not found in model.")

    # ── Fixed-steps-per-frame constants ───────────────────────────────────
    # Chạy đúng SIM_STEPS_PER_FRAME bước vật lý trước mỗi viewer.sync().
    # → mỗi frame render luôn có cùng lượng dịch chuyển → không jitter/jump.
    # 10 bước × 0.002 s = 0.020 s sim/frame  →  target 50 fps real-time.
    SIM_STEPS_PER_FRAME = 10
    TARGET_FRAME_TIME   = SIM_STEPS_PER_FRAME * model.opt.timestep  # 0.020 s

    step_count = 0
    t_frame    = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.lookat[:]  = [1.0, 0.0, 0.53]
        viewer.cam.distance   = 4.5
        viewer.cam.elevation  = -20
        viewer.cam.azimuth    = 160

        while viewer.is_running():

            # ── Luôn chạy đúng SIM_STEPS_PER_FRAME bước vật lý ───────────
            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
                manager.run_conveyor(excluded=push_sched.active_pushes)
                push_sched.step(data.time)
                manager.step(data.time)
                step_count += 1
                if step_count % VISION_EVERY_STEPS == 0:
