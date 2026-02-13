"""
Calibration subsystem for real-time gaze tracking.

This module provides:

- CalibrationManager: stateful calibration controller
- classify_gaze: maps averaged eye position to screen-relative region

Calibration flow:
1. User looks at predefined screen points.
2. Stable gaze is detected via low positional variance.
3. Mean eye position per point is stored.
4. Gaze classification compares live eye position to those anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import cv2


# -----------------------------
# Type Aliases
# -----------------------------

PointDef = Tuple[str, float, float]
"""
(name, tx, ty)
- name: calibration point label
- tx, ty: normalized screen coordinates (0.0 → 1.0)
"""

XY = Tuple[float, float]
"""
(x, y) in pixel coordinates
"""


# -----------------------------
# Configuration Object
# -----------------------------

@dataclass
class CalibrationConfig:
    """
    Configuration for calibration behavior.
    """

    points: List[PointDef]

    stable_seconds: float = 2.5
    """
    Minimum duration user must hold stable gaze before point is accepted.
    """

    std_threshold_px: float = 2.0
    """
    Maximum allowed standard deviation (in pixels)
    for gaze to be considered stable.
    """

    min_samples: int = 10
    """
    Minimum samples required before computing stability.
    """

    buffer_maxlen: int = 90
    """
    Max rolling eye samples stored.
    (~3 seconds at ~30 FPS)
    """


# -----------------------------
# Calibration Manager
# -----------------------------

class CalibrationManager:
    """
    Stateful calibration subsystem.

    Public API:
    - start(now)
    - update(now, avg_eye_xy)
    - draw(overlay)
    - result()
    - is_active()

    This class owns all calibration state.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize calibration manager.

        If no config is provided, a default 5-point calibration is used.
        """
        if config is None:
            config = CalibrationConfig(
                points=[
                    ("center", 0.5, 0.5),
                    ("top_left", 0.05, 0.05),
                    ("top_right", 0.95, 0.05),
                    ("bottom_left", 0.05, 0.95),
                    ("bottom_right", 0.95, 0.95),
                ]
            )

        self.cfg = config

        # Runtime state
        self._active: bool = False
        self._index: int = 0
        self._start_t: float = 0.0
        self._eye_buf: Deque[XY] = deque(maxlen=self.cfg.buffer_maxlen)

        # Raw collected samples per point
        self._data: Dict[str, List[XY]] = {}

        # Final summarized anchors
        self._result: Optional[Dict[str, XY]] = None

        # Per-frame debug state
        self._stable: bool = False
        self._last_std: Optional[Tuple[float, float]] = None


    # -----------------------------
    # Lifecycle
    # -----------------------------

    def start(self, now: float) -> None:
        """
        Begin a new calibration session.
        """
        self._active = True
        self._index = 0
        self._start_t = now
        self._eye_buf.clear()
        self._data = {}
        self._result = None
        self._stable = False
        self._last_std = None


    def is_active(self) -> bool:
        """
        Return whether calibration is currently running.
        """
        return self._active


    # -----------------------------
    # Core Update Logic
    # -----------------------------

    def update(self, now: float, avg_eye_xy: Optional[XY]) -> bool:
        """
        Feed new eye data for the current frame.

        avg_eye_xy:
            (x, y) averaged iris center in pixel coordinates.
            None if face/eyes not detected.

        Returns:
            True if calibration finished during this call.
        """
        if not self._active:
            return False

        name, _, _ = self.cfg.points[self._index]
        self._data.setdefault(name, [])

        # Store sample if valid
        if avg_eye_xy is not None:
            self._eye_buf.append(avg_eye_xy)
            self._data[name].append(avg_eye_xy)

        # Reset stability state
        self._stable = False
        self._last_std = None

        # Compute stability if enough samples collected
        if len(self._eye_buf) >= self.cfg.min_samples:
            xs, ys = zip(*self._eye_buf)

            std_x = float(np.std(xs))
            std_y = float(np.std(ys))

            self._last_std = (std_x, std_y)

            self._stable = (
                std_x < self.cfg.std_threshold_px
                and std_y < self.cfg.std_threshold_px
                and (now - self._start_t) >= self.cfg.stable_seconds
            )

        # Advance to next point if stable
        if self._stable:
            self._index += 1
            self._start_t = now
            self._eye_buf.clear()

            if self._index >= len(self.cfg.points):
                self._active = False
                self._compute_result()
                return True

        return False


    def _compute_result(self) -> None:
        """
        Convert raw eye samples into per-point anchor means.
        """
        result: Dict[str, XY] = {}

        for name, samples in self._data.items():
            if not samples:
                continue

            xs, ys = zip(*samples)
            result[name] = (
                float(np.mean(xs)),
                float(np.mean(ys)),
            )

        self._result = result


    def result(self) -> Optional[Dict[str, XY]]:
        """
        Return calibration anchor dictionary or None
        if calibration has not completed.
        """
        return self._result


    # -----------------------------
    # UI Rendering
    # -----------------------------

    def draw(self, overlay) -> None:
        """
        Draw calibration UI onto current frame (BGR).

        This includes:
        - Target dot
        - Instruction text
        - Stability status
        - Optional debug std display
        """
        if not self._active:
            return

        h, w = overlay.shape[:2]
        name, tx, ty = self.cfg.points[self._index]

        dot_x = int(tx * w)
        dot_y = int(ty * h)

        # Draw target dot
        cv2.circle(overlay, (dot_x, dot_y), 10, (0, 255, 255), -1)

        # Instruction
        cv2.putText(
            overlay,
            f"Calibration: look at the dot ({name})",
            (14, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Status
        status_text = "Captured" if self._stable else "Hold steady..."
        cv2.putText(
            overlay,
            status_text,
            (max(14, dot_x - 60), min(h - 40, dot_y + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self._stable else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Debug: show std values
        if self._last_std is not None:
            std_x, std_y = self._last_std
            cv2.putText(
                overlay,
                f"std: ({std_x:.2f}, {std_y:.2f})  thresh<{self.cfg.std_threshold_px:.2f}",
                (14, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


# -----------------------------
# Gaze Classification
# -----------------------------

def classify_gaze(avg_x: float, avg_y: float, calib: dict, sensitivity: float = 0.5):
    """
    Classify gaze direction based on displacement from calibrated center.

    Parameters
    ----------
    avg_x, avg_y : float
        Current averaged iris position in pixel coordinates.

    calib : dict
        Calibration result dictionary containing:
            {
                "center": (x, y),
                "top_left": (x, y),
                "top_right": (x, y),
                "bottom_left": (x, y),
                "bottom_right": (x, y),
            }

    sensitivity : float
        Fraction of the calibrated eye movement range used as threshold.
        Lower = more sensitive (triggers easier).
        Higher = more strict (requires larger movement).

    Returns
    -------
    str
        One of:
        "on_screen", "off_left", "off_right", "off_up", "off_down"
    """

    # ---- Center anchor ----
    center_x, center_y = calib["center"]

    # ---- Horizontal calibration spread ----
    left_x  = min(calib["top_left"][0],  calib["bottom_left"][0])
    right_x = max(calib["top_right"][0], calib["bottom_right"][0])

    horizontal_range = right_x - left_x

    # ---- Vertical calibration spread ----
    up_y   = min(calib["top_left"][1],  calib["top_right"][1])
    down_y = max(calib["bottom_left"][1], calib["bottom_right"][1])

    vertical_range = down_y - up_y

    # ---- Derive dynamic thresholds ----
    horizontal_threshold = horizontal_range * sensitivity
    vertical_threshold   = vertical_range * sensitivity

    # ---- Compute displacement from center ----
    dx = avg_x - center_x
    dy = avg_y - center_y

    # ---- Classification ----
    if dx < -horizontal_threshold:
        return "off_left"

    if dx > horizontal_threshold:
        return "off_right"

    if dy < -vertical_threshold:
        return "off_up"

    if dy > vertical_threshold:
        return "off_down"

    return "on_screen"