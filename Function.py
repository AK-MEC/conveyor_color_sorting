

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
