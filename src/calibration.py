# src/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import cv2


PointDef = Tuple[str, float, float]          # (name, tx, ty) in normalized coords
XY = Tuple[float, float]                     # (x, y) in pixel coords


@dataclass
class CalibrationConfig:
    points: List[PointDef]
    stable_seconds: float = 2.5
    std_threshold_px: float = 2.0
    min_samples: int = 10
    buffer_maxlen: int = 90                  # ~3s at 30fps


class CalibrationManager:
    """
    Stateful calibration subsystem.
    - start(now): begins calibration
    - update(now, avg_eye_xy): feed eye samples (x,y) in pixel coords
    - draw(overlay): draws dot + instructions
    - result(): returns calibration_result dict or None
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        if config is None:
            config = CalibrationConfig(
                points=[
                    ("center", 0.5, 0.5),
                    ("top_left", 0.1, 0.1),
                    ("top_right", 0.9, 0.1),
                    ("bottom_left", 0.1, 0.9),
                    ("bottom_right", 0.9, 0.9),
                ]
            )
        self.cfg = config

        # runtime state
        self._active: bool = False
        self._index: int = 0
        self._start_t: float = 0.0
        self._eye_buf: Deque[XY] = deque(maxlen=self.cfg.buffer_maxlen)

        # raw + summarized calibration
        self._data: Dict[str, List[XY]] = {}
        self._result: Optional[Dict[str, XY]] = None

        # computed per-frame during calibration (for draw/debug)
        self._stable: bool = False
        self._last_std: Optional[Tuple[float, float]] = None  # (std_x, std_y)

    def start(self, now: float) -> None:
        """Begin a new calibration session."""
        self._active = True
        self._index = 0
        self._start_t = now
        self._eye_buf.clear()
        self._data = {}
        self._result = None
        self._stable = False
        self._last_std = None

    def is_active(self) -> bool:
        return self._active

    def update(self, now: float, avg_eye_xy: Optional[XY]) -> bool:
        """
        Feed new eye data for the current frame.
        - avg_eye_xy is (x,y) pixel coords for averaged iris centers.
        Returns True if calibration just finished on this update.
        """
        if not self._active:
            return False

        name, _, _ = self.cfg.points[self._index]
        self._data.setdefault(name, [])

        if avg_eye_xy is not None:
            self._eye_buf.append(avg_eye_xy)
            self._data[name].append(avg_eye_xy)

        # determine stability
        self._stable = False
        self._last_std = None

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

        # advance on stable
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
        """Summarize raw samples into per-point anchors (means)."""
        result: Dict[str, XY] = {}
        for name, samples in self._data.items():
            if not samples:
                continue
            xs, ys = zip(*samples)
            result[name] = (float(np.mean(xs)), float(np.mean(ys)))
        self._result = result

    def result(self) -> Optional[Dict[str, XY]]:
        """Return calibration_result dict or None."""
        return self._result

    def draw(self, overlay) -> None:
        """
        Draw calibration UI onto overlay.
        Assumes overlay is the current frame (BGR) and uses its size.
        """
        if not self._active:
            return

        h, w = overlay.shape[:2]
        name, tx, ty = self.cfg.points[self._index]

        dot_x = int(tx * w)
        dot_y = int(ty * h)

        # dot
        cv2.circle(overlay, (dot_x, dot_y), 10, (0, 255, 255), -1)

        # instruction
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

        # status
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

        # optional debug: show std
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


def classify_gaze(avg_x: float, avg_y: float, calib: Dict[str, XY]) -> str:
    """
    Very coarse gaze classifier using calibration anchors.
    Expects calib keys: center, top_left, top_right, bottom_left.
    """
    # (cx, cy) not used yet but kept for clarity/future
    _cx, _cy = calib["center"]

    if avg_x < calib["top_left"][0]:
        return "off_left"
    if avg_x > calib["top_right"][0]:
        return "off_right"
    if avg_y < calib["top_left"][1]:
        return "off_up"
    if avg_y > calib["bottom_left"][1]:
        return "off_down"
    return "on_screen"