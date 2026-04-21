import collections
import dataclasses
import queue
import cv2
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH   = os.path.join(SCRIPT_DIR, "conveyor.xml")

N_OBJECTS      = 10
BELT_VELOCITY  = 0.200
SPAWN_INTERVAL = 1.5
SPAWN_X        = 0.10
SPAWN_Y        = 0.00
SPAWN_Z        = 0.556
SPAWN_QUAT     = [1, 0, 0, 0]

BELT_X_MIN  =  0.00
BELT_X_MAX  =  2.00
BELT_Y_MAX  =  0.12
BELT_Z_MIN  =  0.540
BELT_Z_MAX  =  0.640

SETTLE_Z               = 0.25
SETTLE_SPEED           = 0.08
SETTLE_MIN_ACTIVE_TIME = 3.0
SETTLE_Y_PUSH_THRESH   = 0.28

CAMERA_NAME        = "side_camera"
CAMERA_X           = 0.40
CAMERA_DETECT_WIN  = 0.07
VISION_EVERY_STEPS = 5
RENDER_W           = 320
RENDER_H           = 240
MAIN_RENDER_W      = 960
MAIN_RENDER_H      = 600

HSV_RED_LO1    = np.array([  0, 120,  80], np.uint8)
HSV_RED_HI1    = np.array([ 10, 255, 255], np.uint8)
HSV_RED_LO2    = np.array([168, 120,  80], np.uint8)
HSV_RED_HI2    = np.array([180, 255, 255], np.uint8)
HSV_YELLOW_LO  = np.array([ 22, 120,  80], np.uint8)
HSV_YELLOW_HI  = np.array([ 38, 255, 255], np.uint8)
COLOR_MIN_PX   = 60

PUSHER_RED_X      = 0.80
PUSHER_YELLOW_X   = 1.480

YELLOW_DETECT_X   = 1.35
YELLOW_OBJECTS    = frozenset([2, 5, 8])

PUSHER_RED_EXTENDED    = -0.24
PUSHER_YELLOW_EXTENDED = -0.24
PUSHER_RETRACTED  =  0.00
PUSH_HOLD_TIME    =  0.8

PUSH_CLEAR_Y      =  0.22

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

RED_OBJECTS     = frozenset([0, 3, 6, 9])
BLUE_OBJECTS    = frozenset([1, 4, 7])
MANUAL_COOLDOWN = 2.0

class SharedState:
    def __init__(self) -> None:
        self._lock                = threading.Lock()
        self._mode                = 'auto'
        self._speed               = 1.0
        self._spawn_req: str | None = None
        self._cooldown_until_real = 0.0

    @property
    def mode(self) -> str:
        with self._lock: return self._mode

    @mode.setter
    def mode(self, v: str) -> None:
        with self._lock: self._mode = v

    @property
    def speed(self) -> float:
        with self._lock: return self._speed

    @speed.setter
    def speed(self, v: float) -> None:
        with self._lock: self._speed = v

    def request_spawn(self, color: str) -> bool:
        with self._lock:
            if self._spawn_req is None:
                self._spawn_req = color
                return True
            return False

    def consume_spawn(self) -> 'str | None':
        with self._lock:
            c = self._spawn_req
            self._spawn_req = None
            return c

    @property
    def in_cooldown(self) -> bool:
        with self._lock:
            return time.perf_counter() < self._cooldown_until_real

    def set_cooldown(self, duration_real_s: float) -> None:
        with self._lock:
            self._cooldown_until_real = time.perf_counter() + duration_real_s

    def cooldown_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._cooldown_until_real - time.perf_counter())

class CamState:
    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._lookat    = np.array([1.0, 0.0, 0.53])
        self._distance  = 4.5
        self._elevation = -20.0
        self._azimuth   = 160.0

    def rotate(self, dx: float, dy: float) -> None:
        with self._lock:
            self._azimuth   = (self._azimuth - dx * 0.4) % 360
            self._elevation = float(np.clip(self._elevation + dy * 0.4, -89, 89))

    def pan(self, dx: float, dy: float) -> None:
        with self._lock:
            az    = np.radians(self._azimuth)
            scale = self._distance * 0.0015
            right = np.array([np.sin(az), -np.cos(az), 0.0])
            self._lookat -= right * dx * scale
            self._lookat[2] += dy * scale

    def zoom(self, factor: float) -> None:
        with self._lock:
            self._distance = float(np.clip(self._distance * factor, 0.3, 20.0))

    def get_mjv_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        with self._lock:
            cam.lookat[:]  = self._lookat.copy()
            cam.distance   = self._distance
            cam.elevation  = self._elevation
            cam.azimuth    = self._azimuth
        return cam

def detect_color(frame_bgr: np.ndarray) -> str:
    h, w = frame_bgr.shape[:2]
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    roi = frame_bgr[y1:y2, x1:x2]

    if roi.size == 0:
        return 'NONE'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_r1  = cv2.inRange(hsv, HSV_RED_LO1, HSV_RED_HI1)
    mask_r2  = cv2.inRange(hsv, HSV_RED_LO2, HSV_RED_HI2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_yellow = cv2.inRange(hsv, HSV_YELLOW_LO, HSV_YELLOW_HI)

    n_red    = int(cv2.countNonZero(mask_red))
    n_yellow = int(cv2.countNonZero(mask_yellow))

    if n_red >= COLOR_MIN_PX and n_red >= n_yellow:
        return 'R'
    if n_yellow >= COLOR_MIN_PX:
        return 'Y'
    return 'NONE'

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

        def _act(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"Actuator '{name}' not found in model.")
            return aid

        self._act_id: dict[str, int] = {
            'R': _act("act_push_red"),
            'Y': _act("act_push_yellow"),
        }

        self._obj_qpos_adr: list[int] = []
        for i in range(N_OBJECTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"obj{i}_free")
            if jid < 0:
                raise RuntimeError(f"Joint 'obj{i}_free' not found in model.")
            self._obj_qpos_adr.append(model.jnt_qposadr[jid])

        self._pending:   list[_PushEvent]   = []
        self.active_pushes: set[int] = set()
        self._last_fire_sim_time: dict[int, float] = {}

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

        self.state:        list[str]              = ['idle'] * N_OBJECTS
        self.idle_queue:   collections.deque[int] = collections.deque(range(N_OBJECTS))
        self.fallen_queue: collections.deque[int] = collections.deque()
        self.last_spawn:   float                  = -SPAWN_INTERVAL
        self.spawn_time:   dict[int, float]       = {i: 0.0 for i in range(N_OBJECTS)}

        for i in range(N_OBJECTS):
            self._write_pose(i, IDLE_POS[i], SPAWN_QUAT)

        self._shared: 'SharedState | None' = None

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

    def set_shared(self, shared: 'SharedState') -> None:
        self._shared = shared

    def _return_to_idle(self, idx: int) -> None:
        self._write_pose(idx, IDLE_POS[idx], SPAWN_QUAT)
        self.state[idx] = 'idle'

    def manual_spawn_by_color(self, color: str, sim_time: float) -> bool:
        group = {
            'R': RED_OBJECTS,
            'B': BLUE_OBJECTS,
            'Y': YELLOW_OBJECTS,
        }.get(color, frozenset())

        for idx in sorted(group):
            if self.state[idx] in ('idle', 'fallen'):
                try:
                    self.fallen_queue.remove(idx)
                except ValueError:
                    pass
                self._do_spawn(idx, sim_time)
                self.last_spawn = sim_time
                return True
        return False

    def _do_spawn(self, idx: int, sim_time: float) -> None:
        self._write_pose(idx, [SPAWN_X, SPAWN_Y, SPAWN_Z], SPAWN_QUAT)
        va = self.qvel_adr[idx]
        self.data.qvel[va]     = BELT_VELOCITY
        self.data.qvel[va + 1] = 0.0
        self.data.qvel[va + 2] = 0.0
        self.state[idx]       = 'active'
        self.spawn_time[idx]  = sim_time

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

    def step(self, sim_time: float) -> None:
        for i in range(N_OBJECTS):
            if self.state[i] != 'active':
                continue
            if (sim_time - self.spawn_time[i]) < SETTLE_MIN_ACTIVE_TIME:
                continue

            x, y, z = self._read_pos(i)
            speed    = self._read_speed(i)
            off_belt = (x > BELT_X_MAX) or (abs(y) > SETTLE_Y_PUSH_THRESH)

            if off_belt and z < SETTLE_Z and speed < SETTLE_SPEED:
                self.state[i] = 'fallen'
                self.fallen_queue.append(i)

        if self._shared and self._shared.mode == 'manual':
            return

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

def _run_sim_embedded(shared: SharedState,
                      frame_q: queue.Queue,
                      cam_state: CamState,
                      stop_event: threading.Event) -> None:
    if not os.path.isfile(XML_PATH):
        print(f"[SIM ERROR] Model file not found: {XML_PATH}")
        return

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        manager    = ObjectQueueManager(model, data)
        manager.set_shared(shared)
        push_sched = PushScheduler(model, data, manager)

        side_renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if cam_id < 0:
            raise RuntimeError(f"Camera '{CAMERA_NAME}' not found in model.")

        main_renderer = mujoco.Renderer(model, height=MAIN_RENDER_H, width=MAIN_RENDER_W)

        SIM_STEPS_PER_FRAME = 10
        TARGET_FRAME_TIME   = SIM_STEPS_PER_FRAME * model.opt.timestep

        step_count = 0
        t_frame    = time.perf_counter()
        prev_mode  = 'auto'

        while not stop_event.is_set():
            current_mode = shared.mode
            if current_mode != prev_mode:
                if current_mode == 'auto':
                    for i in range(N_OBJECTS):
                        if manager.state[i] == 'idle':
                            manager.state[i] = 'fallen'
                            manager.fallen_queue.append(i)
                prev_mode = current_mode

            speed = shared.speed
            steps_this_frame = max(10, round(SIM_STEPS_PER_FRAME * speed))

            for _ in range(steps_this_frame):
                mujoco.mj_step(model, data)
                manager.run_conveyor(excluded=push_sched.active_pushes)
                push_sched.step(data.time)
                manager.step(data.time)
                step_count += 1
                if step_count % VISION_EVERY_STEPS == 0:
                    _vision_tick(data, manager, push_sched, side_renderer, cam_id)

            color_req = shared.consume_spawn()
            if color_req and shared.mode == 'manual':
                spawned = manager.manual_spawn_by_color(color_req, data.time)
                if spawned:
                    shared.set_cooldown(MANUAL_COOLDOWN / speed)

            try:
                mjv_cam = cam_state.get_mjv_camera()
                main_renderer.update_scene(data, camera=mjv_cam)
                frame = main_renderer.render()
                frame_q.put_nowait(frame.copy())
            except queue.Full:
                pass
            except Exception as ex:
                print(f"[RENDER] {ex}")

            elapsed = time.perf_counter() - t_frame
            if elapsed < TARGET_FRAME_TIME:
                time.sleep(TARGET_FRAME_TIME - elapsed)
            t_frame = time.perf_counter()

        side_renderer.close()
        main_renderer.close()

    except Exception as e:
        print(f"[SIM ERROR] {e}")

def run() -> None:
    if not _PIL_OK:
        raise ImportError(
            "Pillow is required for the embedded viewer.\n"
            "Install it with:  pip install Pillow")

    if not os.path.isfile(XML_PATH):
        raise FileNotFoundError(
            f"Model file not found: {XML_PATH}\n"
            "Place conveyor.xml in the same directory as this script.")

    shared     = SharedState()
    frame_q    : queue.Queue = queue.Queue(maxsize=2)
    cam_state  = CamState()
    stop_event = threading.Event()

    sim_thread = threading.Thread(
        target=_run_sim_embedded,
        args=(shared, frame_q, cam_state, stop_event),
        daemon=True,
    )
    sim_thread.start()

    _launch_embedded_ui(shared, frame_q, cam_state, stop_event)

def _vision_tick(
    data:       mujoco.MjData,
    manager:    ObjectQueueManager,
    push_sched: PushScheduler,
    renderer:   mujoco.Renderer,
    cam_id:     int,
) -> None:
    for i in range(N_OBJECTS):
        if manager.state[i] != 'active':
            continue
        if i in push_sched.active_pushes:
            continue

        x, _, _ = manager._read_pos(i)
        if abs(x - CAMERA_X) > CAMERA_DETECT_WIN:
            continue

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

def _launch_embedded_ui(shared: SharedState,
                        frame_q: queue.Queue,
                        cam_state: CamState,
                        stop_event: threading.Event) -> None:
    BG         = "#1a1a2e"
    BG2        = "#16213e"
    FG         = "#e0e0e0"
    ACCENT     = "#0f3460"
    BTN_AUTO   = "#1b5e20"
    BTN_MAN    = "#bf360c"
    C_BLUE     = "#1565c0"
    C_RED      = "#c62828"
    C_YELLOW   = "#f57f17"
    C_DISABLED = "#424242"

    PANEL_W = 260

    root = tk.Tk()
    root.title("\U0001f3ed  Belt Conveyor Simulation")
    root.configure(bg=BG)
    root.geometry("1240x650")
    root.minsize(800, 480)

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    canvas = tk.Canvas(root, bg="#0a0a0a", cursor="fleur",
                       highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    panel = tk.Frame(root, bg=BG, width=PANEL_W)
    panel.pack(side="right", fill="y")
    panel.pack_propagate(False)

    _photo_ref: list = [None]

    def _update_canvas():
        try:
            frame = frame_q.get_nowait()
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            if cw > 4 and ch > 4:
                img   = Image.fromarray(frame, "RGB").resize(
                            (cw, ch), Image.BILINEAR)
                photo = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor="nw", image=photo)
                _photo_ref[0] = photo
        except queue.Empty:
            pass
        root.after(16, _update_canvas)

    root.after(120, _update_canvas)

    _drag: dict = {"x": 0, "y": 0, "btn": 0}

    def _on_press(event):
        _drag["x"], _drag["y"], _drag["btn"] = event.x, event.y, event.num

    def _on_drag(event):
        dx, dy = event.x - _drag["x"], event.y - _drag["y"]
        _drag["x"], _drag["y"] = event.x, event.y
        if _drag["btn"] == 1:
            cam_state.rotate(dx, dy)
        elif _drag["btn"] == 3:
            cam_state.pan(dx, -dy)

    def _on_scroll(event):
        if event.delta:
            cam_state.zoom(0.9 if event.delta > 0 else 1.1)
        elif event.num == 4:
            cam_state.zoom(0.9)
        elif event.num == 5:
            cam_state.zoom(1.1)

    canvas.bind("<ButtonPress-1>", _on_press)
    canvas.bind("<ButtonPress-3>", _on_press)
    canvas.bind("<B1-Motion>",     _on_drag)
    canvas.bind("<B3-Motion>",     _on_drag)
    canvas.bind("<MouseWheel>",    _on_scroll)
    canvas.bind("<Button-4>",      _on_scroll)
    canvas.bind("<Button-5>",      _on_scroll)

    tk.Label(canvas,
             text="  \U0001f5b1  Drag L: xoay  |  Drag R: pan  |  Scroll: zoom  ",
             bg="#000000", fg="#888888",
             font=("Helvetica", 8)).place(relx=0.0, rely=1.0, anchor="sw", x=0, y=0)

    pad = dict(padx=12, pady=3)

    title_bar = tk.Frame(panel, bg=ACCENT, height=36)
    title_bar.pack(fill="x")
    title_bar.pack_propagate(False)

    tk.Label(title_bar, text="  \U0001f3ed  Conveyor Control",
             bg=ACCENT, fg=FG,
             font=("Helvetica", 10, "bold")).pack(side="left", pady=4, padx=8)

    def sep():
        tk.Frame(panel, bg=ACCENT, height=1).pack(fill="x", padx=0, pady=3)

    sep()

    lbl_mode_status = tk.Label(panel, text="\u25cf Ch\u1ebf \u0111\u1ed9 t\u1ef1 \u0111\u1ed9ng",
                                bg=BG, fg="#66bb6a", font=("Helvetica", 8))
    lbl_mode_status.pack(anchor="w", padx=14, pady=(4, 0))

    def toggle_mode():
        shared.mode = "manual" if shared.mode == "auto" else "auto"

    btn_mode = tk.Button(
        panel, text="AUTO MODE", bg=BTN_AUTO, fg="white",
        font=("Helvetica", 10, "bold"), relief="flat",
        activebackground=BTN_MAN, activeforeground="white",
        width=20, height=1, cursor="hand2",
        command=toggle_mode)
    btn_mode.pack(**pad)

    sep()

    tk.Label(panel, text="\u23f1  T\u1ed1c \u0111\u1ed9 m\u00f4 ph\u1ecfng",
             bg=BG, fg=FG,
             font=("Helvetica", 9, "bold")).pack(anchor="w", padx=14, pady=(4, 0))

    speed_var   = tk.DoubleVar(value=1.0)
    speed_frame = tk.Frame(panel, bg=BG2, bd=0, highlightthickness=1,
                           highlightbackground=ACCENT)
    speed_frame.pack(fill="x", padx=12, pady=4)

    def on_speed_change():
        shared.speed = speed_var.get()

    for label, value in [("1\u00d7", 1.0), ("1.5\u00d7", 1.5), ("2\u00d7", 2.0)]:
        tk.Radiobutton(
            speed_frame, text=label, variable=speed_var, value=value,
            bg=BG2, fg=FG, selectcolor=ACCENT,
            activebackground=BG2, activeforeground=FG,
            font=("Helvetica", 10, "bold"),
            command=on_speed_change,
        ).pack(side="left", padx=10, pady=6)

    sep()

    tk.Label(panel, text="\U0001f3af  Th\u1ea3 v\u1eadt (ch\u1ebf \u0111\u1ed9 th\u1ee7 c\u00f4ng)",
             bg=BG, fg=FG,
             font=("Helvetica", 9, "bold")).pack(anchor="w", padx=14, pady=(4, 0))

    spawn_frame = tk.Frame(panel, bg=BG)
    spawn_frame.pack(fill="x", padx=12, pady=4)

    def make_spawn_cb(color_code: str):
        def cb():
            if shared.mode == "manual" and not shared.in_cooldown:
                shared.request_spawn(color_code)
        return cb

    btn_blue = tk.Button(
        spawn_frame, text="  \U0001f7e6  BLUE  ", bg=C_BLUE, fg="white",
        font=("Helvetica", 10, "bold"), relief="flat",
        activebackground="#1976d2", activeforeground="white",
        width=18, cursor="hand2", command=make_spawn_cb("B"))
    btn_blue.pack(pady=2, fill="x")

    btn_red = tk.Button(
        spawn_frame, text="  \U0001f7e5  RED   ", bg=C_RED, fg="white",
        font=("Helvetica", 10, "bold"), relief="flat",
        activebackground="#d32f2f", activeforeground="white",
        width=18, cursor="hand2", command=make_spawn_cb("R"))
    btn_red.pack(pady=2, fill="x")

    btn_yellow = tk.Button(
        spawn_frame, text="  \U0001f7e8  YELLOW", bg=C_YELLOW, fg="white",
        font=("Helvetica", 10, "bold"), relief="flat",
        activebackground="#f9a825", activeforeground="white",
        width=18, cursor="hand2", command=make_spawn_cb("Y"))
    btn_yellow.pack(pady=2, fill="x")

    lbl_cd = tk.Label(panel, text="", bg=BG, fg="#ffab40",
                      font=("Helvetica", 9, "bold"))
    lbl_cd.pack(pady=(2, 8))

    def refresh():
        mode      = shared.mode
        in_cd     = shared.in_cooldown
        cd_rem    = shared.cooldown_remaining()
        is_manual = (mode == "manual")

        lbl_mode_status.config(
            text="\u25cf Ch\u1ebf \u0111\u1ed9 th\u1ee7 c\u00f4ng" if is_manual
                 else "\u25cf Ch\u1ebf \u0111\u1ed9 t\u1ef1 \u0111\u1ed9ng",
            fg="#ff8a65" if is_manual else "#66bb6a",
        )
        btn_mode.config(
            text="MANUAL MODE" if is_manual else "AUTO MODE",
            bg=BTN_MAN if is_manual else BTN_AUTO,
            activebackground=BTN_AUTO if is_manual else BTN_MAN,
        )

        can_spawn = is_manual and not in_cd
        for btn, normal_bg in [(btn_blue, C_BLUE), (btn_red, C_RED),
                                (btn_yellow, C_YELLOW)]:
            btn.config(
                state="normal" if can_spawn else "disabled",
                bg=normal_bg if can_spawn else C_DISABLED,
            )

        if is_manual and in_cd:
            lbl_cd.config(text=f"\u23f3 Ch\u1edd: {cd_rem:.1f} s")
        elif is_manual:
            lbl_cd.config(text="\u2705 S\u1eb5n s\u00e0ng", fg="#66bb6a")
        else:
            lbl_cd.config(text="")

        root.after(80, refresh)

    refresh()
    root.mainloop()

if __name__ == "__main__":
    run()