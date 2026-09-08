"""
debug_minimal_multiframe.py — Tests whether a TRIVIAL single-block DeltaCNN
module (conv -> BN -> leaky activation, no residuals, no routes) also
diverges on the second consecutive delta step, the same way the full
YOLOv3 does. This isolates whether the bug is in DeltaCNN's core delta
mechanism itself, or specific to how our architecture uses DCAdd /
DCConcatenate / DCUpsamplingNearest2d.
"""

import torch
import torch.nn as nn
import deltacnn

torch.manual_seed(0)
device = "cuda"


class MiniDCBlock(deltacnn.DCModule):
    def __init__(self):
        super().__init__()
        self.sparsify = deltacnn.DCSparsify(delta_threshold=0.0, dilation=15)
        self.conv = deltacnn.DCConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, backend=deltacnn.DCBackend.delta_cudnn)
        self.bn = deltacnn.DCBatchNorm2d(32)
        self.act = deltacnn.DCActivation(activation="leaky")

    def forward(self, x):
        x = self.sparsify(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


def main():
    # Dense reference: same weights, no temporal state, always correct
    conv_dense = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    bn_dense = nn.BatchNorm2d(32).to(device)
    act_dense = nn.LeakyReLU(0.01).to(device)  # matches DCActivation's hardcoded slope
    conv_dense.eval(); bn_dense.eval()

    with torch.no_grad():
        bn_dense.running_mean.copy_(torch.randn(32, device=device) * 0.5)
        bn_dense.running_var.copy_(torch.rand(32, device=device) + 0.5)
        bn_dense.weight.copy_(torch.rand(32, device=device) + 0.5)
        bn_dense.bias.copy_(torch.randn(32, device=device) * 0.3)

    model_dc = MiniDCBlock().to(device, memory_format=torch.channels_last)
    model_dc.eval()
    model_dc.conv.weight.data.copy_(conv_dense.weight.data)
    model_dc.bn.weight.data.copy_(bn_dense.weight.data)
    model_dc.bn.bias.data.copy_(bn_dense.bias.data)
    model_dc.bn.running_mean.data.copy_(bn_dense.running_mean.data)
    model_dc.bn.running_var.data.copy_(bn_dense.running_var.data)
    model_dc.process_filters()

    # Frame 0: random base. Frame 1: SMALL perturbation of frame 0 (mimics a
    # near-static video frame pair). Frame 2: LARGE change (mimics real motion
    # or a big scene change). This tests whether it's delta MAGNITUDE that
    # matters, not which call number it is.
    frame0 = torch.randn(1, 3, 64, 64, device=device)
    frame1 = frame0 + 0.01 * torch.randn(1, 3, 64, 64, device=device)   # small delta
    frame2 = frame0 + 2.0 * torch.randn(1, 3, 64, 64, device=device)    # large delta
    frames = [frame0, frame1, frame2]
    frames_cl = [f.contiguous(memory_format=torch.channels_last) for f in frames]

    with torch.no_grad():
        dc_out0 = model_dc(frames_cl[0])   # frame 0 — full dense (first call)
        dc_out1 = model_dc(frames_cl[1])   # frame 1 — first delta step
        dc_out2 = model_dc(frames_cl[2])   # frame 2 — SECOND delta step (the one we're testing)

        # DCActivation returns a (value, mask) tuple unless dense_out=True was
        # set on the preceding conv — unwrap to get just the value tensor
        dc_out0 = dc_out0[0] if isinstance(dc_out0, (tuple, list)) else dc_out0
        dc_out1 = dc_out1[0] if isinstance(dc_out1, (tuple, list)) else dc_out1
        dc_out2 = dc_out2[0] if isinstance(dc_out2, (tuple, list)) else dc_out2

        # dense reference has no state, so call it fresh on each frame directly
        dense_out0 = act_dense(bn_dense(conv_dense(frames[0])))
        dense_out1 = act_dense(bn_dense(conv_dense(frames[1])))
        dense_out2 = act_dense(bn_dense(conv_dense(frames[2])))

    labels = ["Frame 0 (base, full dense)", "Frame 1 (SMALL delta from frame 0)", "Frame 2 (LARGE delta from frame 1)"]
    for i, (d, c) in enumerate(zip(
        [dense_out0, dense_out1, dense_out2],
        [dc_out0, dc_out1, dc_out2]
    )):
        diff = (d - c).abs()
        print(f"{labels[i]}: max diff = {diff.max().item():.6f}, mean diff = {diff.mean().item():.6f}")


if __name__ == "__main__":
    main()
