

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
