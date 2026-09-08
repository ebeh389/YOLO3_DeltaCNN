# YOLOv3 with Temporal-Sparse (DeltaCNN) Inference

This repository contains a from-scratch PyTorch implementation of YOLOv3, and a conversion of that same trained model into [DeltaCNN](https://github.com/facebookresearch/DeltaCNN)'s temporal-sparse inference framework — which skips recomputation on regions of a video frame that haven't changed since the previous frame.

This is the software (GPU) stage of a larger project whose eventual goal is implementing this same temporal-sparsity idea directly in custom FPGA hardware (via Vitis HLS) for edge deployment. That FPGA work lives in a separate repository; this repository is where the object-detection model itself, and the DeltaCNN conversion, were developed and validated.

## Background and Credit

The base dense YOLOv3 implementation (`model.py`) is built on [Sanna Persson's YOLOv3-PyTorch](https://github.com/SannaPersson/YOLOv3-PyTorch), a compact, from-scratch PyTorch reimplementation of [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf) (Redmon & Farhadi). Credit to the original author for the base model, loss, and dataset-handling code this project builds on.

The temporal-sparsity technique itself is [DeltaCNN](https://github.com/facebookresearch/DeltaCNN) (Parger et al., CVPR 2022) — see [Section 4](#4-an-important-finding-deltacnns-official-library-diverges-from-dense-output) for why this repository also includes an independent, hand-verified reference implementation of the same idea.

## 1. What's in This Repository

| File | Purpose |
|---|---|
| `model.py` | Dense YOLOv3: model definition, loss function, dataset loader, and shared utilities (IoU, NMS, box conversion, plotting, checkpointing). Imported by every other file below. |
| `train_test.py` | Trains the dense model, and/or previews a training sample / runs inference on a single test image. |
| `model_deltacnn.py` | Defines the DeltaCNN-converted model (`DCYOLOv3`) and the weight-transfer logic that copies a trained dense checkpoint into it. |
| `video_inference_deltacnn.py` | Runs the DeltaCNN model over a video file frame-by-frame, draws detections, writes an annotated output video, and reports per-frame latency. |
| `reference_delta_conv.py` | A minimal, hand-written, correctness-first reference implementation of delta-based convolution — see Section 4. |
| `debug_delta_frame.py`, `debug_minimal_multiframe.py` | Diagnostic scripts used to isolate a numerical divergence found in DeltaCNN's official library (Section 4). Not part of the normal pipeline. |
| `checkpoint.pth.tar` | Trained dense-model weights, loadable by both the dense model and (after conversion) the DeltaCNN model. |
| `dataset/convert_dataset.py` | Converts the raw dataset into the image/label/CSV layout `Dataset` in `model.py` expects (see Section 3). |
| `requirements.txt` | Python dependencies (see Section 2). |

## 2. Setup

```bash
git clone <this-repository-url>
cd <this-repository>
pip install -r requirements.txt
```

`requirements.txt` covers everything except DeltaCNN itself, which is a compiled CUDA extension and must be built from source:

```bash
git clone https://github.com/facebookresearch/DeltaCNN.git
cd DeltaCNN
pip install -e .
```

A CUDA-capable GPU is required for both training and DeltaCNN inference (DeltaCNN's sparse kernels are CUDA-only). The dense model alone can run on CPU, but training on CPU is impractically slow.

## 3. Dataset

`Dataset` (in `model.py`) expects, for each of train/validation:
- A directory of images
- A directory of YOLO-format label `.txt` files (one file per image: `class x_center y_center width height`, normalized 0–1, one row per object)
- A CSV file listing image/label filename pairs

Run `dataset/convert_dataset.py` to produce this layout from the raw source dataset. *(If your raw dataset format differs from what this script expects, adjust its input path/format accordingly — see the script itself for its exact expected input.)*

The class list currently used (`train_test.py` / `video_inference_deltacnn.py`) is the 20 PASCAL VOC classes:
`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor`.

## 4. Training the Dense Model

Edit the constants at the top of `train_test.py` if needed (`batch_size`, `learning_rate`, `epochs`, `image_size`, the `ANCHORS` list — these should be re-derived via k-means on your own dataset if you're not using PASCAL VOC), then:

```python
# in train_test.py, __main__:
run_training()          # instead of preview_random_sample() / run_inference_sample()
```

```bash
python train_test.py
```

This trains `YOLOv3` end-to-end with `YOLOLoss`, saving a checkpoint (`checkpoint.pth.tar`) after every epoch when `save_model = True`. Set `load_model = True` to resume from an existing checkpoint rather than training from scratch.

**Inference on a single image** (uses the same file):
```python
# in train_test.py, __main__:
run_inference_sample()
```
Loads `checkpoint.pth.tar`, runs one image from the validation set through the model, applies NMS, and displays the result with `plot_image`.

## 5. Converting to DeltaCNN and Running Video Inference

DeltaCNN conversion is inference-only — the dense model is trained completely normally first (Section 4); only afterward is the trained model converted into a DeltaCNN-structured copy for sparse inference. Concretely, `model_deltacnn.py` mirrors the dense architecture layer-for-layer, mapping each dense module to its DeltaCNN equivalent:

| Dense (`model.py`) | DeltaCNN (`model_deltacnn.py`) |
|---|---|
| `nn.Conv2d` / `nn.BatchNorm2d` / `nn.LeakyReLU` (via `CNNBlock`) | `DCConv2d` / `DCBatchNorm2d` / `DCActivation` (via `DCCNNBlock`) |
| Residual `x + residual` (in `ResidualBlock`) | `DCAdd()` (in `DCResidualBlock`) |
| `torch.cat(...)` (route/skip connections) | `DCConcatenate()` |
| `nn.Upsample` | `DCUpsamplingNearest2d` |
| Final detection-head conv (plain, no BN/activation) | `DCConv2d(..., dense_out=True)` — output must stay dense, not delta-encoded, since it feeds NMS directly |
| *(no dense equivalent)* | `DCSparsify()` at the network's input — converts the incoming dense frame into DeltaCNN's internal sparse representation |

Because every dense layer has a DeltaCNN counterpart with identical parameter shapes, trained weights transfer by direct copy — no re-training or BatchNorm-fusion math needed (`load_trained_weights_into_dc` in `model_deltacnn.py`).

**Run DeltaCNN inference on a video:**
```bash
python video_inference_deltacnn.py --input path/to/video.mp4 --output annotated.mp4
```
Optional flags: `--checkpoint` (default `checkpoint.pth.tar`), `--conf_threshold` (default `0.75`), `--iou_threshold` (default `0.35`), `--sparsity_threshold` (DeltaCNN's change-detection threshold, default `0.0` = most conservative/least aggressive skipping), `--refresh_interval` (frames between forced full-dense recomputes to bound numerical drift, default `30`; see Section 6).

This writes an annotated output video and prints per-frame latency, plus a summary comparing frame 0 (always a full dense computation) against the average of subsequent frames.

## 6. An Important Finding: DeltaCNN's Official Library Diverges From Dense Output

While validating the DeltaCNN conversion, we found that the official `deltacnn` library's delta-computation path **does not exactly match a dense forward pass**, once more than one delta step has occurred:

- `debug_delta_frame.py` compares the full `DCYOLOv3` model's output on frame *N* (reached by feeding it frames 0 through *N* in order, so its internal delta state is real) against a fresh, stateless dense-model call on the same frame *N*. The first delta step (frame 1) matches closely; by the second delta step, a measurable divergence appears.
- `debug_minimal_multiframe.py` reproduces this in the simplest possible case — a single `conv → batchnorm → activation` block, no residuals or routing — confirming the divergence is in DeltaCNN's core delta mechanism itself, not specific to how our architecture uses it.
- In practice (`video_inference_deltacnn.py`), this shows up as drift that accumulates over a long sequence of delta-only frames: near-perfect agreement with the dense model at the first delta step, but substantial divergence after hundreds of frames with no correction. The `--refresh_interval` flag exists specifically to bound this by periodically forcing a full dense recompute.
- `reference_delta_conv.py` is our own independent, hand-written specification of what *correct* delta computation should do: recompute the dense value for changed regions, and hold the previous frame's output exactly for unchanged regions, with no incremental/accumulating state to drift. It is a correctness reference and deliberately not optimized for speed — its purpose is to define the target behavior that a real sparse/tile-skipping implementation (including the eventual FPGA hardware) should match, since the official library was shown not to.

This is a genuine, verified finding from this project, not a usage error — it should be taken into account by anyone building on `deltacnn` for applications where exact numerical correctness over long sequences matters.

## 7. Results

The base dense architecture, trained on PASCAL VOC by the original YOLOv3-PyTorch author, is reported at **78.2 mAP@50** (confidence threshold 0.2, IoU threshold 0.45, NMS) — see the [original repository](https://github.com/SannaPersson/YOLOv3-PyTorch) for that reference result.

*This section should be updated with the specific mAP, training loss curve, and validated DeltaCNN speedup/accuracy numbers actually achieved by `checkpoint.pth.tar` in this repository, once formally evaluated.* The most concrete, already-verified results from this repository specifically are the qualitative findings in Section 6 above: DeltaCNN reproduces the dense model closely for a single delta step, but diverges over longer sequences without periodic correction.

## 8. Relation to the FPGA Project

This repository's role in the broader project is to establish and validate, in software, the object-detection model and the temporal-sparsity technique that a separate FPGA implementation (via Vitis HLS on a Kria KV260) targets. The FPGA work uses YOLOv3-**tiny** (a smaller variant, more suited to edge-FPGA resource budgets) rather than the full YOLOv3 trained here, and implements its own dense baseline first, with the temporal-sparsity extension — informed directly by the correctness findings in Section 6 — planned as the next phase. See the FPGA project's own repository for that work.
