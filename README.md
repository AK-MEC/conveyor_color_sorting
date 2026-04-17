# 🏭 KGF65-KA Belt Conveyor — MuJoCo Color-Sorting Simulation

> A real-time physics simulation of an industrial belt conveyor with automated color-based object sorting, built on **MuJoCo** + **OpenCV**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Dependencies & Installation](#dependencies--installation)
- [Quick Start](#quick-start)
- [File Reference](#file-reference)
  - [conveyor.xml — MuJoCo World Model](#conveyorxml--mujoco-world-model)
  - [Function.py — Simulation Logic](#functionpy--simulation-logic)
- [Constants & Configuration](#constants--configuration)
- [Core Modules](#core-modules)
  - [detect\_color()](#detect_color)
  - [PushScheduler](#pushscheduler)
  - [ObjectQueueManager](#objectqueuemanager)
  - [\_vision\_tick()](#_vision_tick)
  - [run() — Main Loop](#run--main-loop)
- [Physics & Geometry Reference](#physics--geometry-reference)
- [Sorting Logic — End-to-End Flow](#sorting-logic--end-to-end-flow)
- [Known Bug Fixes Implemented](#known-bug-fixes-implemented)
- [Tuning & Customization](#tuning--customization)

---

## Overview

This project simulates a **2-meter industrial belt conveyor** that automatically sorts 60×60×60 mm cubic objects into three color-coded bins:

| Color  | Bin Position       | Detection Method        | Pusher X |
|--------|--------------------|-------------------------|----------|
| 🔴 Red    | X=0.80, Y=−0.50 m | Camera (OpenCV/HSV)     | X=0.80 m |
| 🟡 Yellow | X=1.50, Y=−0.50 m | Ground-truth index lookup | X=1.48 m |
| 🔵 Blue   | X=2.25, Y= 0.00 m | Falls off belt end naturally | — |

The belt runs at **0.200 m/s** in the +X direction. Objects are spawned at the start of the belt, detected mid-transit, and pushed laterally (−Y direction) into the appropriate bin by pneumatic-style servo pushers.

All belt motion is **pure kinematic override** — no tendon/constraint physics; Python injects velocity directly into the MuJoCo `qvel` state each step.

---

## Project Structure

```
.
├── Function.py          # Main simulation: physics loop, vision, scheduling
├── conveyor.xml         # MuJoCo MJCF world model (belt, bins, pushers, camera)
├── simlify.stl          # 3D mesh of the conveyor frame (visual only)
└── belt_stripe.png      # Belt surface texture (green stripe pattern)
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         run()  — Main Loop                       │
│  ┌───────────┐   ┌──────────────┐   ┌───────────────────────┐    │
│  │ mj_step() │──▶│run_conveyor()│──▶│  push_sched.step()   │    │
│  │ (physics) │   │(belt vx inj.)│   │ (fire/retract pushers)│    │
│  └───────────┘   └──────────────┘   └───────────────────────┘    │
│        │                                        │                │
│        ▼                                        ▼                │
│  ┌───────────┐   ┌──────────────────────────────────────────┐    │
│  │manager    │   │         _vision_tick()  (every 5 steps)  │    │
│  │.step()    │   │  ┌─────────────────┐  ┌──────────────┐   │    │
│  │(spawn /   │   │  │ Camera @ X=0.40 │  │ GT @ X=1.35  │   │    │
│  │ settle)   │   │  │ detect_color()  │  │ index lookup │   │    │
│  └───────────┘   │  │ → RED only      │  │ → YELLOW only│   │    │
│                  │  └────────┬────────┘  └──────┬───────┘   │    │
│                  │           └─────────┬─────────┘          │    │
│                  │               push_sched.schedule()      │    │
│                  └──────────────────────────────────────────┘    │
│                                        │                         │
│                              viewer.sync()  (50 fps)             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dependencies & Installation

```bash
pip install mujoco numpy opencv-python
```

| Package       | Purpose                                       |
|---------------|-----------------------------------------------|
| `mujoco`      | Physics engine, renderer, passive viewer      |
| `numpy`       | Array operations for pose/velocity read-write |
| `opencv-python` | BGR→HSV color detection from camera frames |

Python ≥ 3.10 recommended (uses `dict[int, list[float]]` type hints).

---

## Quick Start

```bash
# Place all four files in the same directory, then:
python Function.py
```

A MuJoCo passive viewer will open. The simulation starts immediately — objects spawn at the belt entrance every 1.5 seconds and are sorted automatically.

**Viewer controls** (standard MuJoCo):
- `Left-drag` — rotate camera
- `Right-drag` / scroll — zoom
- `Ctrl+Left-drag` — pan

---

## File Reference

### `conveyor.xml` — MuJoCo World Model

The MJCF (MuJoCo XML Format) file defines the entire simulated world. It uses **no tendons and no equality constraints** — all belt motion is handled in Python.

#### Solver Configuration

```xml
<option gravity="0 0 -9.81"
        timestep="0.002"
        solver="Newton"
        iterations="100"
        tolerance="1e-10"
        cone="pyramidal"
        impratio="10"/>
```

| Parameter    | Value       | Rationale                                              |
|--------------|-------------|--------------------------------------------------------|
| `timestep`   | 0.002 s     | 500 Hz physics; stable for fast contacts (pushers)     |
| `solver`     | Newton      | Most accurate; required for stiff contacts             |
| `iterations` | 100         | High count ensures convergence of belt/object contacts |
| `impratio`   | 10          | Improves constraint stability for box-on-belt sliding  |

#### Assets

| Asset Name      | Type    | Description                                                   |
|-----------------|---------|---------------------------------------------------------------|
| `simlify`       | mesh    | STL frame mesh, scaled 0.001× (mm → m)                        |
| `belt_tex`      | texture | 2D image from `belt_stripe.png`; repeats 20× along belt       |
| `metal_tex`     | texture | Flat gray (built-in) for structural frame parts               |
| `drum_tex`      | texture | Checker pattern for drive/passive drum cylinders              |
| `ground_mat`    | material| Checker grid for ground plane                                 |
| `obj0_mat`…`obj9_mat` | material | Per-object RGBA: pattern R B Y R B Y R B Y R           |

#### Object Color Pattern

Objects 0–9 follow a fixed repeating pattern:

```
Index:  0    1    2    3    4    5    6    7    8    9
Color: RED  BLUE YEL  RED  BLUE YEL  RED  BLUE YEL  RED
```

Yellow objects (indices 2, 5, 8) are hardcoded in `YELLOW_OBJECTS = frozenset([2, 5, 8])`.

#### Geometry (World Frame, Z-up)

```
Z = 0.000  Ground plane
Z = 0.474  Rail top surface
Z = 0.487  Bed top surface
Z = 0.500  Drum axis
Z = 0.524  Belt visual surface
Z = 0.526  Belt physics floor top  ← effective belt top
Z = 0.556  Object center when resting on belt  (= 0.526 + 0.030 half-height)
```

#### Belt Physics Floor

The belt surface collision is a **separate invisible geom** from the visual belt:

```xml
<geom name="belt_floor" type="box"
      size="1.0 0.125 0.002"
      pos="1.0 0 0.524"
      friction="1.4 0.008 0.0006"   <!-- High friction to grip objects -->
      rgba="0 0 0 0"/>              <!-- Invisible -->
```

High friction (`1.4`) prevents objects from sliding laterally on their own.

#### Camera (`side_camera`)

Mounted on a physical body at world position `(0.40, −0.19, 0.70)`, tilted **47° downward** around X to look at the belt surface:

```xml
<body name="camera_body" pos="0.40 -0.19 0.70" euler="47 0 0">
  <camera name="side_camera" fovy="50"/>
</body>
```

This camera is used **only for RED detection**. Its field of view at belt level covers approximately X = 0.33…0.47 m (CAMERA_X ± CAMERA_DETECT_WIN).

#### Pushers

Both pushers are identical in geometry; they differ only in X position and actuator name.

```
Pusher body anchor:  Y = +0.20 m  (outside the +Y rail at Y = +0.138)
Slide joint:         axis = world Y,  range = [−0.45, 0.00] m
Retracted (q=0.00):  Face front at Y ≈ +0.19 m  (clear of belt)
Extended  (q=−0.45): Face front at Y ≈ −0.26 m  (past belt edge at Y=−0.138)
```

| Pusher         | Body X    | Actuator Name      | Targets        |
|----------------|-----------|--------------------|----------------|
| `pusher_red`   | 0.80 m    | `act_push_red`     | RED objects    |
| `pusher_yellow`| 1.480 m   | `act_push_yellow`  | YELLOW objects |

Both use **position servos** (`kp=3000 N/m`) with force limits of ±1000 N.

#### Bins

| Bin         | Body Position    | Color RGBA          | Opens toward    |
|-------------|------------------|---------------------|-----------------|
| `red_bin`   | (0.80, −0.50, 0) | `0.85 0.08 0.08 1`  | +X side (belt)  |
| `blue_bin`  | (2.25,  0.00, 0) | `0.08 0.18 0.82 1`  | −X side (drum)  |
| `yellow_bin`| (1.50, −0.50, 0) | `0.92 0.80 0.00 1`  | +X side (belt)  |

Each bin is constructed from 5 walls: floor + 3 solid walls + 1 half-height gate wall facing the belt.

---

### `Function.py` — Simulation Logic

All runtime logic lives here. The file is structured as:

```
Constants (module-level)
├── Belt / spawn parameters
├── Settle detection thresholds
├── Vision / camera parameters
├── HSV color ranges
├── Pusher geometry constants
└── Initial staging positions

Functions & Classes
├── detect_color(frame_bgr)          — OpenCV color classifier
├── _PushEvent (dataclass)           — Single push action record
├── PushScheduler                    — Pusher timing & control
│   ├── __init__()
│   ├── schedule(color, time, idx, x)
│   ├── _obj_clear_of_belt(ev)
│   └── step(sim_time)
├── ObjectQueueManager               — Object lifecycle (spawn/settle/recycle)
│   ├── __init__()
│   ├── _write_pose()
│   ├── _read_pos()
│   ├── _read_speed()
│   ├── _do_spawn()
│   ├── run_conveyor(excluded)
│   └── step(sim_time)
├── run()                            — Main simulation entry point
└── _vision_tick()                   — Per-step vision subsystem
```

---

## Constants & Configuration

### Belt / Spawn

| Constant         | Value     | Description                                                  |
|------------------|-----------|--------------------------------------------------------------|
| `N_OBJECTS`      | 10        | Total objects in the simulation (recycled in FIFO)           |
| `BELT_VELOCITY`  | 0.200 m/s | Belt surface speed in +X direction                           |
| `SPAWN_INTERVAL` | 1.5 s     | Minimum time between successive object spawns                |
| `SPAWN_X`        | 0.10 m    | X position where objects appear on the belt                  |
| `SPAWN_Y`        | 0.00 m    | Y position at spawn (belt centerline)                        |
| `SPAWN_Z`        | 0.556 m   | Z position at spawn = belt top (0.526) + half-height (0.030) |
| `SPAWN_QUAT`     | [1,0,0,0] | Upright orientation quaternion (no rotation)                 |

### Belt Zone (Kinematic Override Region)

| Constant      | Value  | Description                                                 |
|---------------|--------|-------------------------------------------------------------|
| `BELT_X_MIN`  | 0.00 m | Start of belt (drive drum center)                           |
| `BELT_X_MAX`  | 2.00 m | End of belt (passive drum center); objects free-fall beyond |
| `BELT_Y_MAX`  | 0.12 m | Half-width of kinematic override zone (matches rail gap)    |
| `BELT_Z_MIN`  | 0.540 m| Lower Z bound of belt zone                                  |
| `BELT_Z_MAX`  | 0.640 m| Upper Z bound of belt zone                                  |

### Settle Detection

An object is considered "settled" (ready for recycling) when ALL conditions are met:

| Constant                | Value   | Meaning                                                  |
|-------------------------|---------|----------------------------------------------------------|
| `SETTLE_Z`              | 0.25 m  | Object must be below this Z (has fallen off belt)        |
| `SETTLE_SPEED`          | 0.08 m/s| Object speed must be below this (at rest)                |
| `SETTLE_MIN_ACTIVE_TIME`| 3.0 s  | Minimum time on belt before settle check                  |
| `SETTLE_Y_PUSH_THRESH`  | 0.28 m  | `|Y| > 0.28` means pushed into a bin                     |

### Vision

| Constant            | Value       | Description                                         |
|---------------------|-------------|-----------------------------------------------------|
| `CAMERA_NAME`       | side_camera | MuJoCo camera name in XML                           |
| `CAMERA_X`          | 0.40 m      | World-X position of detection window center         |
| `CAMERA_DETECT_WIN` | 0.07 m      | ±window — object must be within X ∈ [0.33, 0.47]    |
| `VISION_EVERY_STEPS`| 5           | Vision runs every 5 sim steps (= every 10 ms)       |
| `RENDER_W`          | 320 px      | Camera render width                                 |
| `RENDER_H`          | 240 px      | Camera render height                                |

### HSV Color Ranges (OpenCV convention: H=0–180, S/V=0–255)

```python
# Red requires TWO ranges because hue wraps at 0°/180°
HSV_RED_LO1 = [  0, 120,  80]   HSV_RED_HI1 = [ 10, 255, 255]  # H near 0°
HSV_RED_LO2 = [168, 120,  80]   HSV_RED_HI2 = [180, 255, 255]  # H near 180°

HSV_YELLOW_LO = [22, 120, 80]   HSV_YELLOW_HI = [38, 255, 255] # H ≈ 30°

COLOR_MIN_PX = 60   # Minimum qualifying pixels in ROI
```

### Pusher Geometry

| Constant                  | Value     | Meaning                                             |
|---------------------------|-----------|-----------------------------------------------------|
| `PUSHER_RED_X`            | 0.80 m    | World-X center of red pusher body                   |
| `PUSHER_YELLOW_X`         | 1.480 m   | World-X center of yellow pusher body                |
| `YELLOW_DETECT_X`         | 1.35 m    | Ground-truth yellow detection point (not camera)    |
| `PUSHER_RED_EXTENDED`     | −0.24     | `ctrl` value for RED push stroke (q = −0.24 m)      |
| `PUSHER_YELLOW_EXTENDED`  | −0.24     | `ctrl` value for YELLOW push stroke                 |
| `PUSHER_RETRACTED`        | 0.00      | `ctrl` value for both pushers at rest               |
| `PUSH_HOLD_TIME`          | 0.8 s     | Safety timeout: force retract after this duration   |
| `PUSH_CLEAR_Y`            | 0.22 m    | `|Y| > 0.22 m` → object cleared the belt rail       |

**Pusher stroke geometry derivation:**

```
Body anchor Y     = +0.20 m
Face half-thick   =  0.010 m
Object half-size  =  0.030 m

Face front Y      = 0.20 + q − 0.010
Object center Y   = Face front Y + 0.030 = 0.22 + q

At q = −0.24:
  Face front Y   = 0.22 − 0.24 − 0.010 = −0.03  →  −0.03 m (past rail at −0.138)
  Object center  = 0.22 − 0.24          = −0.02  →  gaining momentum toward bin
```

---

## Core Modules

### `detect_color()`

```python
def detect_color(frame_bgr: np.ndarray) -> str:
```

**Purpose**: Classify the color of the object visible in a camera frame.

**Algorithm**:

1. **ROI crop**: Take only the center 60% of the frame (rows 20%–80%, cols 20%–80%). This filters out belt edges, frame geometry, and background.
2. **Color space**: Convert ROI from BGR → HSV (`cv2.cvtColor`).
3. **Red mask**: Create two HSV range masks (for hue wrapping at 0°/180°), combine with `bitwise_or`.
4. **Yellow mask**: Single HSV range mask.
5. **Pixel count**: Count non-zero pixels in each mask (`cv2.countNonZero`).
6. **Decision**:
   - If `n_red ≥ 60` AND `n_red ≥ n_yellow` → return `'R'`
   - If `n_yellow ≥ 60` → return `'Y'`
   - Otherwise → return `'NONE'` (blue object or empty belt)

**Why red uses two hue ranges**: In OpenCV HSV, Hue is mapped 0–180°. Pure red has H ≈ 0 (wraps from 360°). A red object near `H=0` may register pixels at both `H≈0` and `H≈178–180` due to anti-aliasing and lighting variation.

**Return values**:

| Return  | Meaning                                       |
|---------|-----------------------------------------------|
| `'R'`   | Red object detected → schedule red pusher     |
| `'Y'`   | Yellow object detected                        |
| `'NONE'`| Blue object or no object in frame             |

---

### `PushScheduler`

Translates vision detections into timed, physics-correct pusher actuations.

#### `_PushEvent` Dataclass

| Field         | Type    | Description                                                     |
|---------------|---------|-----------------------------------------------------------------|
| `fire_time`   | `float` | Simulation time (seconds) when pusher should extend             |
| `max_retract` | `float` | Safety fallback: force retract at `fire_time + PUSH_HOLD_TIME`  |
| `act_id`      | `int`   | Index into `data.ctrl` for the actuator                         |
| `obj_idx`     | `int`   | Object index currently being pushed                             |
| `qpos_adr`    | `int`   | MuJoCo `qpos` start address for the object (Y is +1)            |
| `color`       | `str`   | `'R'` or `'Y'` — determines stroke distance                     |
| `fired`       | `bool`  | Whether the pusher has extended yet                             |
| `retracted`   | `bool`  | Whether the pusher has returned to rest                         |

#### `__init__()`

Initializes the scheduler by:
- Looking up actuator IDs by name (`act_push_red`, `act_push_yellow`) via `mujoco.mj_name2id`.
- Caching `qpos_adr` for all 10 objects to enable Y-position monitoring during push.
- Setting both pusher controls to `PUSHER_RETRACTED = 0.00`.
- Creating `_last_fire_sim_time: dict[int, float]` for ghost-push prevention.

#### `schedule(color, det_time, obj_idx, obj_x)`

Schedules a push event with three guard conditions:

**Guard 1 — No duplicate pending:**
```python
if any(ev.obj_idx == obj_idx for ev in self._pending):
    return
```
Prevents scheduling the same object twice while it's already in the queue.

**Guard 2 — Yellow ghost-push prevention:**
```python
if obj_idx in self._last_fire_sim_time:
    if self._mgr.spawn_time[obj_idx] <= self._last_fire_sim_time[obj_idx]:
        return  # Object not yet recycled — block re-schedule
```
After a push, the object settles in a bin and eventually gets recycled (re-spawned). Without this guard, the scheduler might re-detect the same object index after it's re-spawned but before the guard's `spawn_time` is updated, causing a "ghost" push on a new object.

**Guard 3 — No actuator overlap:**
```python
for ev in self._pending:
    if ev.act_id == act_id and fire_time <= ev.max_retract:
        return  # Real conflict — pusher still extending
```
Unlike a simple "block if same actuator", this only blocks if the new `fire_time` falls within the existing event's active window. This allows two consecutive same-color objects with non-overlapping windows to both be scheduled.

**Delay calculation:**
```python
delay     = max(0.05, (pusher_x - obj_x) / BELT_VELOCITY)
fire_time = det_time + delay
```
The delay is computed from the object's **actual X position at detection time**, not a fixed offset. This corrects for objects detected slightly early/late in the window.

#### `step(sim_time)`

Called every simulation step. Iterates all pending events:

**Phase 1 — Fire:**
```
if not fired AND sim_time ≥ fire_time:
    data.ctrl[act_id] = PUSHER_EXTENDED
    active_pushes.add(obj_idx)
    _last_fire_sim_time[obj_idx] = sim_time
    ev.fired = True
```

**Phase 2 — Smart retract:**
```
if fired AND not retracted:
    min_hold_ok  = sim_time ≥ fire_time + 0.3 s   (pusher physically extends in ~0.3 s)
    obj_cleared  = min_hold_ok AND |Y_obj| > PUSH_CLEAR_Y
    time_expired = sim_time ≥ max_retract

    if obj_cleared OR time_expired:
        data.ctrl[act_id] = PUSHER_RETRACTED
        active_pushes.discard(obj_idx)
        ev.retracted = True
```

**Why smart retract?** A fixed-time retract can pull the object back onto the belt if the pusher retracts before the object has cleared the rail. Monitoring `|Y|` ensures the pusher holds until the object has definitely left the belt zone.

**Phase 3 — Cleanup:**
Completed events (retracted) are removed from `_pending`.

---

### `ObjectQueueManager`

Manages the FIFO infinite conveyor loop: spawn → travel → sort → settle → recycle.

#### State Machine (per object)

```
  'idle'  ──(dequeued at startup)──▶  'active'  ──(settled)──▶  'fallen'
     ▲                                                               │
     │                                                               │
     └───────────────────────────────(re-spawned)───────────────────┘
```

#### Queues

| Queue          | Type                   | Description                                      |
|----------------|------------------------|--------------------------------------------------|
| `idle_queue`   | `deque(range(10))`     | Populated at startup; drained once at sim start  |
| `fallen_queue` | `deque()`              | FIFO of settled objects awaiting re-spawn        |

#### `_write_pose(idx, pos, quat)`

Directly writes position and orientation into `data.qpos` and zeros `data.qvel`. Used for both spawning and initial staging.

#### `_read_pos(idx)` / `_read_speed(idx)`

Read object position `(x, y, z)` and scalar speed `|v|` from `data.qpos` / `data.qvel` using cached addresses.

#### `_do_spawn(idx, sim_time)`

Places object at `(SPAWN_X, SPAWN_Y, SPAWN_Z)` with initial velocity `vx = BELT_VELOCITY`. Updates `state[idx] = 'active'` and records `spawn_time[idx]`.

#### `run_conveyor(excluded)`

The core belt mechanism. For every `'active'` object not in `excluded`:

1. Read object position.
2. Skip if `x > BELT_X_MAX` (past end — free-fall, physics takes over).
3. Check if object is in the kinematic zone:
   ```python
   on_belt = (BELT_X_MIN ≤ x ≤ BELT_X_MAX
              AND |y| ≤ BELT_Y_MAX
              AND BELT_Z_MIN ≤ z ≤ BELT_Z_MAX)
   ```
4. If on belt: force `vx = BELT_VELOCITY`, zero all other velocity components, reset quaternion to `SPAWN_QUAT` (keeps objects upright).

**`excluded` parameter**: Objects currently being pushed by a servo are excluded. Without this, the belt override would cancel the pusher's Y-velocity impulse every step, preventing the object from being pushed.

#### `step(sim_time)`

**Settle detection** (evaluated per active object):
```python
off_belt = (x > BELT_X_MAX) OR (|y| > SETTLE_Y_PUSH_THRESH)
settled  = off_belt AND z < SETTLE_Z AND speed < SETTLE_SPEED
         AND (sim_time - spawn_time) ≥ SETTLE_MIN_ACTIVE_TIME
```

When settled: `state[i] = 'fallen'`, append to `fallen_queue`.

**Spawn timer**:
```
if (sim_time - last_spawn) ≥ SPAWN_INTERVAL:
    if idle_queue is non-empty:  spawn from idle_queue
    elif fallen_queue non-empty: spawn from fallen_queue (FIFO)
```

Phase 1 (startup): drains `idle_queue` sequentially — each object spawns 1.5 s after the previous.
Phase 2 (steady state): recycles fallen objects.

---

### `_vision_tick()`

Called every `VISION_EVERY_STEPS = 5` simulation steps (every 10 ms real time).

#### RED Detection (Camera-based)

```python
for i in range(N_OBJECTS):
    if state[i] != 'active': continue
    if i in push_sched.active_pushes: continue
    x, _, _ = manager._read_pos(i)
    if abs(x - CAMERA_X) > CAMERA_DETECT_WIN: continue

    # Render and classify
    renderer.update_scene(data, camera=cam_id)
    frame_rgb = renderer.render()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    color = detect_color(frame_bgr)
    if color == 'R':
        push_sched.schedule('R', data.time, i, obj_x=x)
    break  # Only one render per vision tick
```

Key decisions:
- Only renders when an object is within `CAMERA_DETECT_WIN` of `CAMERA_X`. This avoids expensive renders when no object is near the camera.
- Breaks after the first object found — one render per tick for performance.
- Skips objects already being pushed.

#### YELLOW Detection (Ground-truth)

```python
for i in YELLOW_OBJECTS:  # frozenset {2, 5, 8}
    if state[i] != 'active': continue
    if i in push_sched.active_pushes: continue
    x, _, _ = manager._read_pos(i)
    if abs(x - YELLOW_DETECT_X) > CAMERA_DETECT_WIN: continue
    push_sched.schedule('Y', data.time, i, obj_x=x)
    break
```

**Why not use the camera for yellow?**
The camera is at X=0.40 m. The yellow pusher is at X=1.48 m. Using camera detection would require a delay of:
```
(1.480 − 0.40) / 0.200 = 5.40 seconds
```
Over 5 seconds of delay accumulates significant position error. Detecting at X=1.35 m reduces delay to:
```
(1.480 − 1.35) / 0.200 = 0.65 seconds
```
This makes yellow pusher timing far more reliable. Since yellow objects are at fixed indices (2, 5, 8), no image processing is needed.

---

### `run()` — Main Loop

```python
SIM_STEPS_PER_FRAME = 10          
TARGET_FRAME_TIME   = 10 × 0.002  
```

**Fixed-steps-per-frame design**: Rather than running as many physics steps as possible per frame (which would make simulation speed dependent on CPU load), exactly 10 steps are run per frame. This gives:
- Consistent simulation speed regardless of vision processing time.
- Smooth viewer rendering (no "jumps" from variable step counts).

**Main loop structure:**

```python
for _ in range(SIM_STEPS_PER_FRAME):
    mujoco.mj_step(model, data)           
    manager.run_conveyor(                  
        excluded=push_sched.active_pushes)
    push_sched.step(data.time)           
    manager.step(data.time)               
    step_count += 1
    if step_count % VISION_EVERY_STEPS == 0:
        _vision_tick(...)                  
viewer.sync()                            

# Pace to 50 fps
elapsed = time.perf_counter() - t_frame
if elapsed < TARGET_FRAME_TIME:
    time.sleep(TARGET_FRAME_TIME - elapsed)
```

**Viewer camera initial position:**
```python
viewer.cam.lookat   = [1.0, 0.0, 0.53]  
viewer.cam.distance = 4.5              
viewer.cam.elevation = -20             
viewer.cam.azimuth  = 160             
```

---

## Physics & Geometry Reference

### Contact Parameters (Default Geom)

| Parameter  | Value               | Effect                                            |
|------------|---------------------|---------------------------------------------------|
| `condim`   | 4                   | 4D friction cone (X+Y translation + spin)         |
| `friction` | `0.8 0.008 0.0005`  | Slide, torsion, rolling friction coefficients     |
| `solimp`   | `0.98 0.999 0.001 0.5 2` | Constraint impedance (softness/stiffness)    |
| `solref`   | `0.010 1`           | Constraint reference (10 ms time constant)        |

### Object Physical Properties

```xml
<default class="flat_obj">
  <geom type="box" size="0.030 0.030 0.030"   <!-- 60×60×60 mm cube -->
        density="600"                         
        friction="0.75 0.006 0.0004"/>
```

Mass = 600 kg/m³ × (0.06 m)³ = 600 × 0.000216 = **0.130 kg** (130 g) per object.

### Belt Floor vs Belt Visual

Two separate geoms occupy nearly the same space:
- `belt_visual`: visual texture, `contype=0` (no collision) — scrolling stripe appearance
- `belt_floor`: invisible box, `contype=1` (collision enabled) — actual physics surface

### Pusher Face Properties

```xml
<default class="pusher_face">
  <geom friction="0.3 0.003 0.0001"   <!-- Low friction: object slides off cleanly -->
        condim="3"/>                   
```

Low friction on the pusher face prevents the object from "sticking" to the pusher as it retracts, which could drag the object back toward the belt.

---

## Sorting Logic — End-to-End Flow

### RED Object Journey

```
t=0.0 s   Object spawns at X=0.10, vx=0.200 m/s
t=1.5 s   Object reaches X ≈ 0.40 m (CAMERA_X)
           → _vision_tick: render camera, detect_color() returns 'R'
           → push_sched.schedule('R', t=1.5, obj_x=0.40)
           → delay = (0.80 − 0.40) / 0.200 = 2.0 s
           → fire_time = 3.5 s

t=3.5 s   Object reaches X ≈ 0.80 m (PUSHER_RED_X)
           → data.ctrl[act_push_red] = −0.24 (extend)
           → Object gets pushed in −Y direction

t≈3.8 s   Object |Y| > 0.22 m → obj_clear_of_belt() = True
           → data.ctrl[act_push_red] = 0.00 (retract)
           → Object falls into RED BIN at (0.80, −0.50)

t≈5.0 s   Object settles in bin: z < 0.25, speed < 0.08
           → state = 'fallen', appended to fallen_queue
```

### YELLOW Object Journey

```
t=0.0 s   Object spawns at X=0.10, vx=0.200 m/s
           (No camera detection — camera only handles RED)

t≈6.25 s  Object reaches X = 1.35 m (YELLOW_DETECT_X)
           → _vision_tick: index in YELLOW_OBJECTS={2,5,8} → schedule directly
           → push_sched.schedule('Y', t=6.25, obj_x=1.35)
           → delay = (1.480 − 1.35) / 0.200 = 0.65 s
           → fire_time = 6.90 s

t=6.90 s  Object reaches X ≈ 1.48 m (PUSHER_YELLOW_X)
           → data.ctrl[act_push_yellow] = −0.24 (extend)
           → Object pushed in −Y direction

t≈7.2 s   Object clears belt → retract → falls into YELLOW BIN at (1.50, −0.50)
```

### BLUE Object Journey

```
No pusher → belt carries it all the way to X=2.00 m (passive drum)
Object falls off belt end → lands in BLUE BIN at (2.25, 0.00) under gravity
```

---

## Known Bug Fixes Implemented

### 1. Yellow Ghost Push (Guard 2)

**Problem**: After a yellow object was pushed into the bin, it settled and was re-spawned as a new belt cycle. The `_last_fire_sim_time` check was absent, so the scheduler could fire on the new spawn before the recycled object's `spawn_time` was recorded, causing an unintended push on a fresh object.

**Fix**: Added `_last_fire_sim_time[obj_idx]` tracking. A re-schedule is only allowed if `spawn_time[obj_idx] > last_fire_sim_time[obj_idx]` — i.e., the object has been genuinely recycled since the last push.

### 2. Red Miss (Guard 3 → overlap-only block)

**Problem**: Earlier version blocked any new RED schedule if `act_push_red` had any pending event, regardless of timing. When two RED objects traveled close together, the second would be blocked even if the first pusher had already retracted.

**Fix**: Guard 3 now only blocks if the new `fire_time` falls within the existing event's `[fire_time, max_retract]` window — a true actuator conflict. Non-overlapping events for the same actuator are allowed.

### 3. Smart Retract (Y-monitoring vs. fixed timer)

**Problem**: Fixed-timer retract (e.g., always retract after 0.4 s) sometimes retracted while the object was still in the pusher's path, pulling it back onto the belt or leaving it stranded between belt and bin.

**Fix**: Retract only when `|Y_object| > PUSH_CLEAR_Y = 0.22 m`, confirming the object has physically cleared the belt rail. A 0.3 s minimum hold time after `fire_time` ensures the pusher has actually extended before checking.

### 4. Yellow Detection Accuracy (Ground-Truth at X=1.35)

**Problem**: Using the camera at X=0.40 for yellow detection required a ~5.4 s delay computation. Even small velocity estimation errors multiplied over 5.4 s caused the pusher to fire too early or too late.

**Fix**: Yellow objects are detected at X=1.35 m (130 mm before the pusher), reducing delay to 0.65 s. Since yellow object indices are fixed and known (`frozenset([2, 5, 8])`), no camera render is needed.

---

## Tuning & Customization

### Change Belt Speed

```python
BELT_VELOCITY = 0.200   
```
Also adjust `SPAWN_INTERVAL` to maintain spacing between objects.

### Add More Objects

```python
N_OBJECTS = 15   
```
Add corresponding entries in `IDLE_POS` and XML `<body>` / `<joint>` definitions.

### Adjust Pusher Stroke

```python
PUSHER_RED_EXTENDED    = -0.30   
PUSHER_YELLOW_EXTENDED = -0.30
```
Range is limited to `[-0.45, 0.0]` by the joint's `range` attribute in XML.

### Change Color Pattern

Edit the `rgba` attribute of `obj0_mat` through `obj9_mat` in `conveyor.xml`, then update `YELLOW_OBJECTS` in `Function.py` to match the new yellow indices.

### Adjust Detection Sensitivity

```python
COLOR_MIN_PX = 60    #
```

### Simulation Speed

```python
SIM_STEPS_PER_FRAME = 10   
                            
```
