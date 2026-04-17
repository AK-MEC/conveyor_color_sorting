import collections
import dataclasses
import cv2
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# Paths 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH   = os.path.join(SCRIPT_DIR, "conveyor.xml")

# Belt / spawn constants 
N_OBJECTS      = 10
BELT_VELOCITY  = 0.200      
SPAWN_INTERVAL = 1.5      
SPAWN_X        = 0.10       
SPAWN_Y        = 0.00        
SPAWN_Z        = 0.556       
SPAWN_QUAT     = [1, 0, 0, 0]

# Belt-zone bounds for kinematic override
BELT_X_MIN  =  0.00
BELT_X_MAX  =  2.00
BELT_Y_MAX  =  0.12
BELT_Z_MIN  =  0.540
BELT_Z_MAX  =  0.640

#  Settle detection 
SETTLE_Z               = 0.25   
SETTLE_SPEED           = 0.08   
SETTLE_MIN_ACTIVE_TIME = 3.0   
SETTLE_Y_PUSH_THRESH   = 0.28 

# Vision 
CAMERA_NAME        = "side_camera"
CAMERA_X           = 0.40    
CAMERA_DETECT_WIN  = 0.07   
VISION_EVERY_STEPS = 5       
RENDER_W           = 320
RENDER_H           = 240

HSV_RED_LO1    = np.array([  0, 120,  80], np.uint8)
HSV_RED_HI1    = np.array([ 10, 255, 255], np.uint8)
HSV_RED_LO2    = np.array([168, 120,  80], np.uint8)   
HSV_RED_HI2    = np.array([180, 255, 255], np.uint8)
HSV_YELLOW_LO  = np.array([ 22, 120,  80], np.uint8)
HSV_YELLOW_HI  = np.array([ 38, 255, 255], np.uint8)
COLOR_MIN_PX   = 60  

# Pusher constants 
PUSHER_RED_X      = 0.80
PUSHER_YELLOW_X   = 1.480  
YELLOW_DETECT_X   = 1.35
YELLOW_OBJECTS    = frozenset([2, 5, 8]) 
PUSHER_RED_EXTENDED    = -0.24
PUSHER_YELLOW_EXTENDED = -0.24 
PUSHER_RETRACTED  =  0.00
PUSH_HOLD_TIME    =  0.8   
PUSH_CLEAR_Y      =  0.22 

# Initial staging positions (Phase-1 only) 
_Z_IDLE = 0.050
IDLE_POS: dict[int, list[float]] = {
    0: [0.7625, -0.4375, _Z_IDLE],  
    3: [0.8375, -0.4375, _Z_IDLE],
    6: [0.7625, -0.3625, _Z_IDLE],
    9: [0.8375, -0.3625, _Z_IDLE],
    1: [2.25,  -0.075,   _Z_IDLE], 
    4: [2.25,   0.000,   _Z_IDLE],
    7: [2.25,   0.075,   _Z_IDLE],
    2: [1.425, -0.400,   _Z_IDLE], 
    5: [1.500, -0.400,   _Z_IDLE],
    8: [1.575, -0.400,   _Z_IDLE],
}
#  detect_color
def detect_color(frame_bgr: np.ndarray) -> str:
    h, w = frame_bgr.shape[:2]

    # ROI: centre 60 % of frame in both dimensions
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 'NONE'
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Red mask
    mask_r1  = cv2.inRange(hsv, HSV_RED_LO1, HSV_RED_HI1)
    mask_r2  = cv2.inRange(hsv, HSV_RED_LO2, HSV_RED_HI2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # ── Yellow mask 
    mask_yellow = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)
    n_red    = int(cv2.countNonZero(mask_red))
    n_yellow = int(cv2.countNonZero(mask_yellow))
    if n_red >= COLOR_MIN_PX and n_red >= n_yellow:
        return 'R'
    if n_yellow >= COLOR_MIN_PX:
        return 'Y'
    return 'NONE'

#  PushScheduler
@dataclasses.dataclass
class _PushEvent:
    fire_time:   float   
    max_retract: float  
    act_id:      int    
    obj_idx:     int     
    qpos_adr:    int     
    color:       str   
    fired:       bool = False
    retracted:   bool = False

class PushScheduler:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 manager: "ObjectQueueManager") -> None:
        self.model   = model
        self.data    = data
        self._mgr    = manager   

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
        self._last_fire_sim_time: dict[int, float] = {}

        # Ensure both pushers start retracted
        for aid in self._act_id.values():
            data.ctrl[aid] = PUSHER_RETRACTED
    def schedule(self, color: str, det_time: float, obj_idx: int,
                 obj_x: float) -> None:
        if color not in ('R', 'Y'):
            return   
        act_id = self._act_id[color]
        if any(ev.obj_idx == obj_idx for ev in self._pending):
            return
        if obj_idx in self._last_fire_sim_time:
            if self._mgr.spawn_time[obj_idx] <= self._last_fire_sim_time[obj_idx]:
                return  
        pusher_x  = PUSHER_RED_X if color == 'R' else PUSHER_YELLOW_X
        delay     = max(0.05, (pusher_x - obj_x) / BELT_VELOCITY)
        fire_time = det_time + delay
        max_ret   = fire_time + PUSH_HOLD_TIME   
        for ev in self._pending:
            if ev.act_id == act_id and fire_time <= ev.max_retract:
                return   
        self._pending.append(_PushEvent(
            fire_time   = fire_time,
            max_retract = max_ret,
            act_id      = act_id,
            obj_idx     = obj_idx,
            qpos_adr    = self._obj_qpos_adr[obj_idx],
            color       = color,
        ))

    def _obj_clear_of_belt(self, ev: _PushEvent) -> bool:
        y = float(self.data.qpos[ev.qpos_adr + 1])
        return abs(y) > PUSH_CLEAR_Y
    def step(self, sim_time: float) -> None:
        done = []
        for ev in self._pending:
            if not ev.fired and sim_time >= ev.fire_time:
                stroke = PUSHER_RED_EXTENDED if ev.color == 'R' else PUSHER_YELLOW_EXTENDED
                self.data.ctrl[ev.act_id] = stroke
                self.active_pushes.add(ev.obj_idx)
                self._last_fire_sim_time[ev.obj_idx] = sim_time
                ev.fired = True
            if ev.fired and not ev.retracted:
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

#  ObjectQueueManager
class ObjectQueueManager:
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

    # Low-level helpers 
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

    # Spawn 
    def _do_spawn(self, idx: int, sim_time: float) -> None:
        self._write_pose(idx, [SPAWN_X, SPAWN_Y, SPAWN_Z], SPAWN_QUAT)
        va = self.qvel_adr[idx]
        self.data.qvel[va]     = BELT_VELOCITY
        self.data.qvel[va + 1] = 0.0
        self.data.qvel[va + 2] = 0.0
        self.state[idx]       = 'active'
        self.spawn_time[idx]  = sim_time

    # belt kinematic override 
    def run_conveyor(self, excluded: set[int] | None = None) -> None:
        excluded = excluded or set()
        for i in range(N_OBJECTS):
            if self.state[i] != 'active':
                continue
            if i in excluded:
                continue  
            qa      = self.qpos_adr[i]
            x, y, z = (float(self.data.qpos[qa]),
                       float(self.data.qpos[qa + 1]),
                       float(self.data.qpos[qa + 2]))
            if x > BELT_X_MAX:
                continue   
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

    # Settle detection + FIFO spawn 
    def step(self, sim_time: float) -> None:
        # ── Settle detection 
        for i in range(N_OBJECTS):
            if self.state[i] != 'active':
                continue
            if (sim_time - self.spawn_time[i]) < SETTLE_MIN_ACTIVE_TIME:
                continue
            x, y, z = self._read_pos(i)
            speed    = self._read_speed(i)

            # Off-belt
            off_belt = (x > BELT_X_MAX) or (abs(y) > SETTLE_Y_PUSH_THRESH)
            if off_belt and z < SETTLE_Z and speed < SETTLE_SPEED:
                self.state[i] = 'fallen'
                self.fallen_queue.append(i)

        # Spawn timer 
        if (sim_time - self.last_spawn) < SPAWN_INTERVAL:
            return
        if self.idle_queue:
            idx = self.idle_queue.popleft()
            self._do_spawn(idx, sim_time)
            self.last_spawn = sim_time
        elif self.fallen_queue:
            idx = self.fallen_queue.popleft()  
            self._do_spawn(idx, sim_time)
            self.last_spawn = sim_time

#  Main simulation loop
def run() -> None:
    if not os.path.isfile(XML_PATH):
        raise FileNotFoundError(
            f"Model file not found: {XML_PATH}\n"
            "Place conveyor.xml in the same directory as this script.")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Subsystem initialisation 
    manager    = ObjectQueueManager(model, data)
    push_sched = PushScheduler(model, data, manager)  
    renderer   = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera '{CAMERA_NAME}' not found in model.")
    SIM_STEPS_PER_FRAME = 10
    TARGET_FRAME_TIME   = SIM_STEPS_PER_FRAME * model.opt.timestep  
    step_count = 0
    t_frame    = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:]  = [1.0, 0.0, 0.53]
        viewer.cam.distance   = 4.5
        viewer.cam.elevation  = -20
        viewer.cam.azimuth    = 160
        while viewer.is_running():
            for _ in range(SIM_STEPS_PER_FRAME):
                mujoco.mj_step(model, data)
                manager.run_conveyor(excluded=push_sched.active_pushes)
                push_sched.step(data.time)
                manager.step(data.time)
                step_count += 1
                if step_count % VISION_EVERY_STEPS == 0:
                    _vision_tick(data, manager, push_sched, renderer, cam_id)
            # ── Render 
            viewer.sync()

            elapsed = time.perf_counter() - t_frame
            if elapsed < TARGET_FRAME_TIME:
                time.sleep(TARGET_FRAME_TIME - elapsed)
            t_frame = time.perf_counter()
    renderer.close()
def _vision_tick(
    data:       mujoco.MjData,
    manager:    ObjectQueueManager,
    push_sched: PushScheduler,
    renderer:   mujoco.Renderer,
    cam_id:     int,
) -> None:

    # ── CAMERA: RED detection only 
    for i in range(N_OBJECTS):
        if manager.state[i] != 'active':
            continue
        if i in push_sched.active_pushes:
            continue
        x, _, _ = manager._read_pos(i)
        if abs(x - CAMERA_X) > CAMERA_DETECT_WIN:
            continue

        # Render va classify 
        renderer.update_scene(data, camera=cam_id)
        frame_rgb = renderer.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        color = detect_color(frame_bgr)
        if color == 'R':
            push_sched.schedule('R', data.time, i, obj_x=x)
        break   
    for i in range(N_OBJECTS):
        if i not in YELLOW_OBJECTS:
            continue
        if manager.state[i] != 'active':
            continue
        if i in push_sched.active_pushes:
            continue
        x, _, _ = manager._read_pos(i)
        if abs(x - YELLOW_DETECT_X) > CAMERA_DETECT_WIN:
            continue
        push_sched.schedule('Y', data.time, i, obj_x=x)
        break
if __name__ == "__main__":
    run()