import numpy as np

# Open Source Computer Vision Library
import cv2

# deque : effiently append frames to the end, and remove frames from the start
from collections import deque

# dataclass : a structured dictionary for settings
from dataclasses import dataclass

# face detection
import mediapipe as mp

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


SMOOTH_ALPHA = 0.4  # 0.0–1.0 (higher = more responsive, less smooth)


def main() -> None:

    print("STARTED")
    print("Opening camera...")

    cfg = AppConfig() # creates an instance of AppConfig dataclass
    cap = cv2.VideoCapture(cfg.camera_index) # open connection to camera device

    print("cap.isOpened =", cap.isOpened())

    # Reference face mesh and drawing utilities
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    CONFIDENCE_THRESHOLD = 0.5

    LEFT_IRIS_IDX = [474, 475, 476, 477]
    RIGHT_IRIS_IDX = [469, 470, 471, 472]

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,   # needed for iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

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

    # Iris draw helper
    def draw_iris(overlay, landmarks, indices, w, h, color=(255, 0, 0)):
        xs, ys = [], []
        for i in indices:
            lm = landmarks.landmark[i]
            x, y = int(lm.x * w), int(lm.y * h)
            xs.append(x)
            ys.append(y)
            cv2.circle(overlay, (x, y), 1, color, -1)

        cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
        return cx, cy

    prev_face_landmarks = None

    quality_events = []
    quality_segment_active = False
    quality_segment_start = None

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

        # Reorder CV frame colours from BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Check whether exactly one face was detected by Face Mesh
        face_detected = (
            results.multi_face_landmarks is not None 
            and len(results.multi_face_landmarks) == 1 
        )

        tracking_confidence = 1.0 if face_detected else 0.0

        can_score = (
            face_detected
            and tracking_confidence >= CONFIDENCE_THRESHOLD
        )

        # Log "bad quality" segment
        if not can_score:
            if not quality_segment_active:
                quality_segment_active = True
                quality_segment_start = now
        else:
            if quality_segment_active:
                quality_events.append({
                    "type": "quality_gate",
                    "start": quality_segment_start,
                    "end": now,
                    "reason": "face_missing_or_low_confidence",
                })
                quality_segment_active = False
                quality_segment_start = None

        # Overlay: FPS + buffer info
        overlay = frame_small.copy()

        # Reset face landmarks if continuity breaks
        if not face_detected:
            prev_face_landmarks = None

        if face_detected:
            current_landmarks = results.multi_face_landmarks[0]

            if prev_face_landmarks is None:
                # First frame: no smoothing yet
                prev_face_landmarks = current_landmarks
                face_landmarks = current_landmarks
            else:
                # EMA smoothing
                for i, lm in enumerate(current_landmarks.landmark):
                    prev = prev_face_landmarks.landmark[i]
                    prev.x = SMOOTH_ALPHA * lm.x + (1 - SMOOTH_ALPHA) * prev.x
                    prev.y = SMOOTH_ALPHA * lm.y + (1 - SMOOTH_ALPHA) * prev.y

                face_landmarks = prev_face_landmarks

            h, w = overlay.shape[:2]

            # Face bounding box from landmarks
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]

            x_min, x_max = int(min(xs) * w), int(max(xs) * w)
            y_min, y_max = int(min(ys) * h), int(max(ys) * h)

            cv2.rectangle(
                overlay,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 255),  # yellow
                2,
            )

            left_iris_center = draw_iris(
                overlay,
                face_landmarks,
                LEFT_IRIS_IDX,
                w,
                h,
            )

            right_iris_center = draw_iris(
                overlay,
                face_landmarks,
                RIGHT_IRIS_IDX,
                w,
                h,
            )

            # Draw full face mesh (debug-first)
            mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )

        info_lines = [
            f"FPS (smoothed): {fps_ema:5.1f}",
            f"Buffer entries: {len(buf)}",
            f"Buffer span:    {buf.age_span_seconds():4.1f}s (target {cfg.buffer_seconds:.0f}s)",
            f"Tracking OK: {can_score}",
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
                0.6,
                (255, 255, 255),
                1,
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