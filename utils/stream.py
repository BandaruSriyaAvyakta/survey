import cv2
import time
import threading
from ultralytics import YOLO
import numpy as np
import io

# Shared stats dict
stats = {
    "frames_processed": 0,
    "detections_total": 0,
    "avg_confidence": 0.0,
    "last_frame_detections": 0,
    "fps": 0.0,
    "action_counts": {},  # Tracks counts of all detected classes (actions and objects)
    "lock": threading.Lock()
}


def _update_stats(num_dets, avg_conf, frame_time, current_frame_actions):
    with stats["lock"]:
        stats["frames_processed"] = stats.get("frames_processed", 0) + 1
        stats["detections_total"] = stats.get("detections_total", 0) + num_dets

        # moving average of avg_confidence
        prev_frames = max(stats["frames_processed"] - 1, 0)
        prev_avg = stats.get("avg_confidence", 0.0)
        stats["avg_confidence"] = ((prev_avg * prev_frames) + avg_conf) / (prev_frames + 1)
        stats["last_frame_detections"] = num_dets

        # fps estimation
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            prev_fps = stats.get("fps", 0.0)
            stats["fps"] = 0.85 * prev_fps + 0.15 * current_fps if prev_fps else current_fps

        # UPDATE: Tally all detected class names (actions/objects) for the current frame
        action_tally = {}
        for action in current_frame_actions:
            action_tally[action] = action_tally.get(action, 0) + 1
        stats["action_counts"] = action_tally


def frame_generator_for_source(model: YOLO, source="cam", file_path=None, conf=0.35):
    """
    Yields MJPEG frames after running YOLO on them, reading all class labels.
    """
    if source == "file":
        cap = cv2.VideoCapture(file_path)
    else:
        cap = cv2.VideoCapture(0)  # webcam

    class_names = model.names

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()

        results = model.predict(source=frame, conf=conf, verbose=False)
        r = results[0]

        boxes = getattr(r, "boxes", None)
        num_dets = 0
        avg_conf = 0.0
        current_frame_actions = []

        if boxes is not None and len(boxes) > 0:
            num_dets = len(boxes)
            confidences = []
            for b in boxes:
                conf_val = float(getattr(b, "conf", 0.0))
                confidences.append(conf_val)

                # Extract the class ID and map to the name
                if b.cls is not None and len(b.cls) > 0:
                    cls_id = int(b.cls[0])
                    action_name = class_names.get(cls_id, "unknown_class")  # Capture all class names (objects/actions)
                    current_frame_actions.append(action_name)

            avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # Draw annotations on frame (r.plot() uses the class names automatically)
        try:
            annotated = r.plot()
        except Exception:
            annotated = frame

        t1 = time.time()
        frame_time = t1 - t0

        _update_stats(num_dets, avg_conf, frame_time, current_frame_actions)

        # encode as JPEG
        ret2, buffer = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        # MJPEG multipart
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()
    # reset stats when stream ends
    with stats["lock"]:
        stats["frames_processed"] = 0
        stats["detections_total"] = 0
        stats["avg_confidence"] = 0.0
        stats["last_frame_detections"] = 0
        stats["fps"] = 0.0
        stats["action_counts"] = {}