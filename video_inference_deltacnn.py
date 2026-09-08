"""
video_inference_deltacnn.py — Run a DeltaCNN-converted YOLOv3 model over a
video file, draw detections on each frame, write an annotated output video,
and print per-frame latency (frame 0 = full dense compute, later frames
should be faster on a mostly-static scene, showing DeltaCNN's temporal
sparsity in action).

Usage:
    python video_inference_deltacnn.py --input surveillance.mp4 --output annotated.mp4
"""

import argparse
import time

import cv2
import numpy as np
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import deltacnn

from model import YOLOv3, load_checkpoint, convert_cells_to_bboxes, nms
from model_deltacnn import DCYOLOv3, load_trained_weights_into_dc

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# Config — keep in sync with train_test.py
# --------------------------------------------------------------------------

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

CLASS_LABELS = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

IMAGE_SIZE = 416
GRID_SIZES = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

# Fixed colors per class for consistent-looking boxes across frames
np.random.seed(0)
BOX_COLORS = [tuple(int(c) for c in np.random.randint(0, 255, 3)) for _ in CLASS_LABELS]

frame_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def build_dc_model(checkpoint_file, learning_rate=1e-4):
    """Loads the trained dense YOLOv3, builds+loads the DeltaCNN equivalent,
    and returns the DC model ready for streaming inference."""
    model = YOLOv3(num_classes=len(CLASS_LABELS)).to(device, memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    load_checkpoint(checkpoint_file, model, optimizer, learning_rate, device)
    model.eval()

    dc_model = DCYOLOv3(num_classes=len(CLASS_LABELS)).to(device, memory_format=torch.channels_last)
    dc_model.eval()
    load_trained_weights_into_dc(model, dc_model)
    dc_model.process_filters()

    return dc_model


# --------------------------------------------------------------------------
# Per-frame processing
# --------------------------------------------------------------------------

def preprocess_frame(frame_bgr):
    """cv2 frame (BGR, HxWx3 uint8) -> normalized tensor ready for the model."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    augmented = frame_transform(image=frame_rgb)["image"]  # CxHxW, normalized
    x = augmented.unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)
    return x


def decode_boxes(outputs, scaled_anchors, conf_threshold=0.75, iou_threshold=0.35):
    """Decodes an already-computed model output tuple into NMS'd boxes.
    Use this when you already have `outputs = model(x)` and don't want to
    re-run the model (e.g. when testing a stateful model like dc_model,
    where calling it again would consume another "frame")."""
    boxes = []
    for i in range(3):
        anchor = scaled_anchors[i]
        S = outputs[i].shape[2]
        boxes += convert_cells_to_bboxes(outputs[i], anchor, s=S, is_predictions=True)[0]

    boxes = nms(boxes, iou_threshold=iou_threshold, threshold=conf_threshold)
    return boxes


def run_detection(dc_model, x, scaled_anchors, conf_threshold=0.75, iou_threshold=0.35):
    """Runs one frame through the model and returns NMS'd boxes:
    [class_id, score, x_center, y_center, width, height] (all normalized 0-1)."""
    with torch.no_grad():
        outputs = dc_model(x)
    return decode_boxes(outputs, scaled_anchors, conf_threshold, iou_threshold)


def draw_boxes(frame_416_rgb_uint8, boxes):
    """Draws boxes (normalized coords) on a 416x416 RGB uint8 frame, in place."""
    h, w = frame_416_rgb_uint8.shape[:2]
    for box in boxes:
        class_id, score = int(box[0]), box[1]
        x_c, y_c, bw, bh = box[2:]

        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        color = BOX_COLORS[class_id]
        cv2.rectangle(frame_416_rgb_uint8, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_LABELS[class_id]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_416_rgb_uint8, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_416_rgb_uint8, label, (x1 + 2, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame_416_rgb_uint8


def tensor_to_display_frame(x):
    """Undo normalization/ToTensorV2 to get a drawable 416x416 uint8 RGB frame."""
    img = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # HxWxC, float, 0-1 range (Normalize divides by 255)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------
# Main video loop
# --------------------------------------------------------------------------

def process_video(input_path, output_path, checkpoint_file="checkpoint.pth.tar",
                   conf_threshold=0.75, iou_threshold=0.35, sparsity_threshold=0.0,
                   refresh_interval=30):
    """refresh_interval: call dc_model.reset_layers() every N frames to force
    a full dense recompute, bounding numerical drift that otherwise
    accumulates over long sequences of delta-only frames (confirmed via
    testing: near-perfect match at frame 1, large divergence by frame 300
    with no refresh). Set to 0 to disable (not recommended for long videos)."""
    deltacnn.DCThreshold.t_default = sparsity_threshold

    dc_model = build_dc_model(checkpoint_file)

    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (IMAGE_SIZE, IMAGE_SIZE)
    )

    frame_idx = 0
    latencies = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if refresh_interval > 0 and frame_idx % refresh_interval == 0 and frame_idx > 0:
            dc_model.reset_layers()  # periodic full-dense refresh to bound drift

        x = preprocess_frame(frame_bgr)

        torch.cuda.synchronize()
        t0 = time.time()
        boxes = run_detection(dc_model, x, scaled_anchors, conf_threshold, iou_threshold)
        torch.cuda.synchronize()
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)
        print(f"frame {frame_idx}: {latency_ms:.2f} ms, {len(boxes)} detections")

        display_frame = tensor_to_display_frame(x)
        display_frame = draw_boxes(display_frame, boxes)
        out_writer.write(cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    cap.release()
    out_writer.release()

    if latencies:
        print("\n--- Summary ---")
        print(f"Frame 0 (full dense compute): {latencies[0]:.2f} ms")
        if len(latencies) > 1:
            avg_rest = sum(latencies[1:]) / len(latencies[1:])
            print(f"Average of remaining {len(latencies) - 1} frames: {avg_rest:.2f} ms")
            print(f"Speedup vs frame 0: {latencies[0] / avg_rest:.2f}x")
    print(f"\nSaved annotated video to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="annotated_output.mp4", help="Path to write annotated output video")
    parser.add_argument("--checkpoint", default="checkpoint.pth.tar")
    parser.add_argument("--conf_threshold", type=float, default=0.75)
    parser.add_argument("--iou_threshold", type=float, default=0.35)
    parser.add_argument("--sparsity_threshold", type=float, default=0.0,
                         help="deltacnn.DCThreshold.t_default — higher = more aggressive skipping")
    parser.add_argument("--refresh_interval", type=int, default=30,
                         help="Force a full dense recompute every N frames to bound drift (0 = disable)")
    args = parser.parse_args()

    process_video(
        args.input, args.output, args.checkpoint,
        args.conf_threshold, args.iou_threshold, args.sparsity_threshold, args.refresh_interval
    )
