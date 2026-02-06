import numpy as np

# Open Source Computer Vision Library
import cv2

# deque : effiently append frames to the end, and remove frames from the start
from collections import deque

# dataclass : a structured dictionary for settings
from dataclasses import dataclass

import time


@dataclass # adds stuff to AppConfig
class AppConfig:
    camera_index: int = 0

    # Target display size (resizing happens every frame)
    display_width: int = 720

    # Rolling buffer config
    buffer_seconds: float = 10.0           # keep last N seconds of entries
    max_fps_assumption: int = 30           # used to cap memory with maxlen

    # FPS smoothing (via ema = Exponential Moving Average)
    fps_ema_alpha: float = 0.15            # higher = more reactive


class RollingBuffer:
    """
    Keeps the last N seconds of (timestamp, frame) entries.
    Uses a deque with a maxlen cap (memory safety) plus time-based trimming (correctness).
    """
    def __init__(self, buffer_seconds: float, maxlen: int):
        self.buffer_seconds = buffer_seconds # how much history is maintained
        self.frames = deque(maxlen=maxlen)  # bounded queue that automatically drops the oldest entry if it gets too big

    # Append new frames
    def push(self, t: float, frame: np.ndarray) -> None:
        self.frames.append((t, frame))
        self._trim_old(t)

    # Remove old ƒrames
    def _trim_old(self, now: float) -> None:
        cutoff = now - self.buffer_seconds
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()

    def __len__(self) -> int:
        return len(self.frames)

    # Sanity checks
    def age_span_seconds(self) -> float:
        # Don't bother computing time spans unless at least 2 frames
        if len(self.frames) < 2:
            return 0.0
        # Output is used to prevent premature calculations / trash early outputs on run
        return self.frames[-1][0] - self.frames[0][0]


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    """
    Resize frames to target width while preserving aspect ratio. 
    """
    h, w = frame.shape[:2] # grab h and w of current frame
    if w == target_width:
        return frame
    scale = target_width / float(w)
    target_height = int(round(h * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA) #resize frame


def main() -> None:

    print("STARTED")
    print("Opening camera...")

    cfg = AppConfig() # creates an instance of AppConfig dataclass
    cap = cv2.VideoCapture(cfg.camera_index) # open connection to camera device

    print("cap.isOpened =", cap.isOpened())

    # if camera don't open stop program immediately
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try a different camera_index or check macOS Camera permissions."
        )

    # Ask camera for a reasonable resolution (not guaranteed to be honored)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Rolling buffer: cap by seconds * assumed fps (memory safety)
    maxlen = int(cfg.buffer_seconds * cfg.max_fps_assumption) + 5
    buf = RollingBuffer(buffer_seconds=cfg.buffer_seconds, maxlen=maxlen)

    prev_t = None
    fps_ema = 0.0

    # Window Display Settings
    window_name = "Real-Time Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read() # "ok" returns True if CV returns a frame
        if not ok or frame is None:
            # If the camera occasionally fails, don't crash; just keep trying.
            # If this happens constantly, permissions or camera selection is wrong.
            time.sleep(0.02)
            continue
        frame = cv2.flip(frame, 1)

        # Initialize and generate now's time in seconds
        now = time.perf_counter()

        # FPS calculation (EMA-smoothed)
        if prev_t is not None:
            dt = max(1e-6, now - prev_t)
            inst_fps = 1.0 / dt
            fps_ema = (cfg.fps_ema_alpha * inst_fps) + ((1.0 - cfg.fps_ema_alpha) * fps_ema)
        prev_t = now

        # Resize for performance and consistent UI
        frame_small = resize_keep_aspect(frame, cfg.display_width)

        # Push into rolling buffer (store resized frames for now)
        buf.push(now, frame_small)

        # Overlay: FPS + buffer info
        overlay = frame_small.copy()
        info_lines = [
            f"FPS (smoothed): {fps_ema:5.1f}",
            f"Buffer entries: {len(buf)}",
            f"Buffer span:    {buf.age_span_seconds():4.1f}s (target {cfg.buffer_seconds:.0f}s)",
            "Quit: q or ESC",
        ]

        y = 28 # starting vertical position
        # HUD Renderer (draw text on the frame)
        for line in info_lines:
            cv2.putText(
                overlay,
                line,
                (14, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28

        cv2.imshow(window_name, overlay)

        # standard OpenCV boilerplate stuff
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = ESC
            break

    cap.release() 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()