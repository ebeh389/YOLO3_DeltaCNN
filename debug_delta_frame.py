"""
debug_delta_frame.py — Isolated test of DeltaCNN's delta (2nd-frame-onward)
computation path, separate from the first-frame path already validated.

Compares: dc_model's output on frame 1, AFTER having already processed
frame 0 (so it takes the real delta path) — against a fresh, stateless
dense-model call on frame 1 (always correct, no temporal state).

Run this once, check the printed diffs, then delete this file — it's a
diagnostic, not part of the pipeline.
"""

import cv2
import torch
import torch.optim as optim

from model import YOLOv3, load_checkpoint
from video_inference_deltacnn import build_dc_model, preprocess_frame, decode_boxes

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_FILE = "checkpoint.pth.tar"
LEARNING_RATE = 1e-4
VIDEO_PATH = "sample.mp4"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
IMAGE_SIZE = 416
GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]


def main():
    # Dense reference model (stateless — always correct)
    model = YOLOv3(num_classes=20).to(device, memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    load_checkpoint(CHECKPOINT_FILE, model, optimizer, LEARNING_RATE, device)
    model.eval()

    # DeltaCNN model (stateful — this is what we're testing)
    dc_model = build_dc_model(CHECKPOINT_FILE, LEARNING_RATE)

    TARGET_FRAME = 2  # testing the SECOND delta step specifically (frame 1 already confirmed correct)

    cap = cv2.VideoCapture(VIDEO_PATH)

    dense_out_target = None
    dc_out_target = None
    frame_idx = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_idx > TARGET_FRAME:
                break

            x = preprocess_frame(frame_bgr)

            # dc_model MUST see every frame in order — its delta state depends
            # on the full history, not just the target frame in isolation
            dc_out = dc_model(x)

            if frame_idx == TARGET_FRAME:
                dc_out_target = dc_out
                # dense model has no temporal state, so it's fine to call it
                # only on the one frame we actually care about
                dense_out_target = model(x)

            frame_idx += 1

    cap.release()
    assert dc_out_target is not None, f"Video had fewer than {TARGET_FRAME + 1} frames"

    print(f"--- Frame {TARGET_FRAME} diff (after replaying full history into dc_model) ---")
    for i in range(3):
        diff = (dense_out_target[i] - dc_out_target[i]).abs().max()
        print(f"Scale {i} max diff: {diff.item()}")

    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    dense_boxes = decode_boxes(dense_out_target, scaled_anchors, conf_threshold=0.75, iou_threshold=0.35)
    dc_boxes = decode_boxes(dc_out_target, scaled_anchors, conf_threshold=0.75, iou_threshold=0.35)

    print(f"\nDense model: {len(dense_boxes)} boxes")
    print(f"DC model: {len(dc_boxes)} boxes")


if __name__ == "__main__":
    main()
