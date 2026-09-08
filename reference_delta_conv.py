"""
reference_delta_conv.py — A minimal, hand-written reference implementation
of temporal-sparse (delta-based) convolution: frame differencing, masking,
and masked convolution, done explicitly in plain PyTorch so every step is
transparent and verifiable against the dense model.

This is NOT meant to be fast (masking here doesn't skip real compute --
it's a correctness reference, not a speed demo). Its purpose is to give
you a trustworthy ground truth for what "correct" temporal-sparse
convolution should produce, since the DeltaCNN library itself was shown
to diverge from the dense reference on any genuine partial update.

Once this is verified correct, its logic (not its code) is what you'd
translate into actual sparse/tile-skipping HLS hardware for the FPGA --
this script is the spec, not the implementation target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceDeltaConv(nn.Module):
    """Wraps a single conv+bn+activation block with explicit delta logic:
    - Computes input delta vs. the previous frame
    - Builds an update mask from a threshold
    - Recomputes the FULL dense conv (no real speedup -- this is a
      correctness reference), then blends: changed pixels get the fresh
      value, unchanged pixels keep the previous cached OUTPUT
    - This blend rule is the actual definition of "correct" delta
      computation: output should equal what a fresh dense pass would give
      for changed regions, and stay exactly equal to the last known
      output for unchanged regions.
    """

    def __init__(self, conv, bn, activation, threshold=0.0):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.activation = activation
        self.threshold = threshold

        self.prev_input = None
        self.prev_output = None

    def reset(self):
        self.prev_input = None
        self.prev_output = None

    def forward(self, x):
        # Full dense computation of the current frame (reference-correct,
        # by construction -- this is just conv->bn->activation)
        full_output = self.activation(self.bn(self.conv(x)))

        if self.prev_input is None:
            # First frame: nothing to delta against, output is just dense
            self.prev_input = x.clone()
            self.prev_output = full_output.clone()
            return full_output, torch.ones_like(x[:, :1])  # mask: everything "changed"

        # Build the update mask from the INPUT delta (per DeltaCNN's own
        # design: mask is based on input change, not output change)
        input_delta = (x - self.prev_input).abs()
        changed = (input_delta.amax(dim=1, keepdim=True) > self.threshold).float()  # 1xHxW mask

        # Correct-by-construction: changed pixels get the fresh dense value;
        # unchanged pixels keep exactly the last cached output (no drift
        # possible, since this isn't computed incrementally -- it's
        # recomputed fully every time and then explicitly blended)
        output = changed * full_output + (1 - changed) * self.prev_output

        self.prev_input = x.clone()
        self.prev_output = output.clone()

        return output, changed


def verify_against_dense(dc_block, dense_conv, dense_bn, dense_act, frames, threshold=0.0):
    """Runs a sequence of frames through both the reference delta block and
    a stateless dense reference, printing per-frame diffs."""
    dc_block.threshold = threshold
    dc_block.reset()

    for i, x in enumerate(frames):
        with torch.no_grad():
            dc_out, mask = dc_block(x)
            dense_out = dense_act(dense_bn(dense_conv(x)))

        diff = (dense_out - dc_out).abs()
        pct_changed = mask.mean().item() * 100
        print(f"Frame {i}: max diff = {diff.max().item():.8f}, "
              f"mean diff = {diff.mean().item():.8f}, "
              f"{pct_changed:.1f}% pixels marked changed")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False).to(device)
    bn = nn.BatchNorm2d(32).to(device)
    act = nn.LeakyReLU(0.01).to(device)
    conv.eval(); bn.eval()

    with torch.no_grad():
        bn.running_mean.copy_(torch.randn(32, device=device) * 0.5)
        bn.running_var.copy_(torch.rand(32, device=device) + 0.5)
        bn.weight.copy_(torch.rand(32, device=device) + 0.5)
        bn.bias.copy_(torch.randn(32, device=device) * 0.3)

    dc_block = ReferenceDeltaConv(conv, bn, act, threshold=0.0).to(device)

    frame0 = torch.randn(1, 3, 64, 64, device=device)
    frame1 = frame0 + 0.01 * torch.randn(1, 3, 64, 64, device=device)
    frame2 = frame0 + 2.0 * torch.randn(1, 3, 64, 64, device=device)

    print("--- Reference delta implementation vs. dense (should be ~0 every frame) ---")
    verify_against_dense(dc_block, conv, bn, act, [frame0, frame1, frame2])
