

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

