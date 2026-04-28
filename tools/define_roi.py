from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

import config


def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    window = "Define ROI - click vertices, 's'=save, 'r'=reset, 'q'=quit"
    cv2.namedWindow(window)
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
            print(f"Added ({x}, {y}); total={len(points)}")

    cv2.setMouseCallback(window, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(points) >= 1:
            for p in points:
                cv2.circle(frame, p, 5, (0, 255, 0), -1)
            if len(points) >= 2:
                cv2.polylines(frame, [np.array(points, dtype=np.int32)],
                              isClosed=len(points) >= 3, color=(0, 255, 0), thickness=2)

        cv2.putText(frame, f"Vertices: {len(points)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            points.clear()
            print("Reset.")
        if key == ord('s'):
            if len(points) < 3:
                print("Need at least 3 vertices to save.")
                continue
            _save_to_config(points)
            print(f"Saved {len(points)} vertices to {Path('config.py').resolve()}")
            break

    cap.release()
    cv2.destroyAllWindows()


def _save_to_config(points: list[tuple[int, int]]) -> None:
    cfg_path = Path(__file__).resolve().parent.parent / "config.py"
    text = cfg_path.read_text()

    formatted = "ROI_POLYGON = [\n" + "".join(
        f"    ({x}, {y}),\n" for x, y in points
    ) + "]"

    new_text, n_subs = re.subn(
        r"ROI_POLYGON\s*=\s*\[[^\]]*\]",
        formatted,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n_subs == 0:
        new_text = text + "\n\n" + formatted + "\n"

    cfg_path.write_text(new_text)


if __name__ == "__main__":
    main()

