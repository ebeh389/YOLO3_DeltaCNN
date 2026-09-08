import deltacnn
import torch
import torch.nn as nn
import torch.optim as optim

from model import YOLOv3, load_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device is {device}')


# --------------------------------------------------------------------------
# DeltaCNN model definition — conv, bn, activation kept as SEPARATE layers,
# matching the official DeltaCNN mobilenet_deltacnn.py reference exactly.
# --------------------------------------------------------------------------


class DCCNNBlock(deltacnn.DCModule):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = deltacnn.DCConv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = deltacnn.DCBatchNorm2d(out_channels)
        self.activation = deltacnn.DCActivation(activation="leaky")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DCResidualBlock(deltacnn.DCModule):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.use_residual = use_residual
        self.blocks = nn.ModuleList()
        self.adders = nn.ModuleList()
        for _ in range(num_repeats):
            self.blocks.append(nn.ModuleList([
                deltacnn.DCConv2d(channels, channels // 2, kernel_size=1, bias=True),
                deltacnn.DCBatchNorm2d(channels // 2),
                deltacnn.DCActivation(activation="leaky"),
                deltacnn.DCConv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                deltacnn.DCBatchNorm2d(channels),
                deltacnn.DCActivation(activation="leaky"),
            ]))
            self.adders.append(deltacnn.DCAdd())  # unique instance per repeat

    def forward(self, x):
        for (conv1, bn1, act1, conv2, bn2, act2), add in zip(self.blocks, self.adders):
            residual = x
            x = conv1(x); x = bn1(x); x = act1(x)
            x = conv2(x); x = bn2(x); x = act2(x)
            if self.use_residual:
                x = add(x, residual)
        return x


class DCScalePrediction(deltacnn.DCModule):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = deltacnn.DCConv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = deltacnn.DCBatchNorm2d(2 * in_channels)
        self.act1 = deltacnn.DCActivation(activation="leaky")
        # final conv: no BN, no activation in the dense original -> stays exactly the same here
        self.conv2 = deltacnn.DCConv2d(2 * in_channels, (num_classes + 5) * 3, kernel_size=1, dense_out=True)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.act1(x)
        output = self.conv2(x)  # already dense due to dense_out=True
        output = output.view(output.size(0), 3, self.num_classes + 5, output.size(2), output.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output


class DCYOLOv3(deltacnn.DCModule):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.sparsify = deltacnn.DCSparsify()

        self.b1 = DCCNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.b2 = DCCNNBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.r1 = DCResidualBlock(64, num_repeats=1)
        self.b3 = DCCNNBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.r2 = DCResidualBlock(128, num_repeats=2)
        self.b4 = DCCNNBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.r3 = DCResidualBlock(256, num_repeats=8)   # route 1 (256-ch)
        self.b5 = DCCNNBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.r4 = DCResidualBlock(512, num_repeats=8)   # route 2 (512-ch)
        self.b6 = DCCNNBlock(512, 1024, kernel_size=3, stride=2, padding=1)
        self.r5 = DCResidualBlock(1024, num_repeats=4)

        self.b7 = DCCNNBlock(1024, 512, kernel_size=1)
        self.b8 = DCCNNBlock(512, 1024, kernel_size=3, padding=1)
        self.r6 = DCResidualBlock(1024, use_residual=False, num_repeats=1)
        self.b9 = DCCNNBlock(1024, 512, kernel_size=1)
        self.pred1 = DCScalePrediction(512, num_classes)

        self.b10 = DCCNNBlock(512, 256, kernel_size=1)
        self.up1 = deltacnn.DCUpsamplingNearest2d(scale_factor=2)
        self.cat1 = deltacnn.DCConcatenate()

        self.b11 = DCCNNBlock(768, 256, kernel_size=1)
        self.b12 = DCCNNBlock(256, 512, kernel_size=3, padding=1)
        self.r7 = DCResidualBlock(512, use_residual=False, num_repeats=1)
        self.b13 = DCCNNBlock(512, 256, kernel_size=1)
        self.pred2 = DCScalePrediction(256, num_classes)

        self.b14 = DCCNNBlock(256, 128, kernel_size=1)
        self.up2 = deltacnn.DCUpsamplingNearest2d(scale_factor=2)
        self.cat2 = deltacnn.DCConcatenate()

        self.b15 = DCCNNBlock(384, 128, kernel_size=1)
        self.b16 = DCCNNBlock(128, 256, kernel_size=3, padding=1)
        self.r8 = DCResidualBlock(256, use_residual=False, num_repeats=1)
        self.b17 = DCCNNBlock(256, 128, kernel_size=1)
        self.pred3 = DCScalePrediction(128, num_classes)

    def forward(self, x):
        x = self.sparsify(x)
        x = self.b1(x); x = self.b2(x); x = self.r1(x)
        x = self.b3(x); x = self.r2(x)
        x = self.b4(x); route1 = self.r3(x)
        x = self.b5(route1); route2 = self.r4(x)
        x = self.b6(route2); x = self.r5(x)

        x = self.b7(x); x = self.b8(x); x = self.r6(x); x = self.b9(x)
        out1 = self.pred1(x)

        x = self.b10(x); x = self.up1(x); x = self.cat1(x, route2)
        x = self.b11(x); x = self.b12(x); x = self.r7(x); x = self.b13(x)
        out2 = self.pred2(x)

        x = self.b14(x); x = self.up2(x); x = self.cat2(x, route1)
        x = self.b15(x); x = self.b16(x); x = self.r8(x); x = self.b17(x)
        out3 = self.pred3(x)

        return out1, out2, out3


# --------------------------------------------------------------------------
# Weight transfer — plain direct copies now, no BN-fusion math needed.
# --------------------------------------------------------------------------

def copy_bn(dense_bn, dc_bn):
    dc_bn.weight.data.copy_(dense_bn.weight.data)
    dc_bn.bias.data.copy_(dense_bn.bias.data)
    dc_bn.running_mean.data.copy_(dense_bn.running_mean.data)
    dc_bn.running_var.data.copy_(dense_bn.running_var.data)


def copy_cnnblock(dense_block, dc_block):
    dc_block.conv.weight.data.copy_(dense_block.conv.weight.data)
    copy_bn(dense_block.bn, dc_block.bn)


def copy_residualblock(dense_block, dc_block):
    for i, seq in enumerate(dense_block.layers):
        # seq = nn.Sequential(conv1, bn1, leakyrelu, conv2, bn2, leakyrelu)
        conv1, bn1 = seq[0], seq[1]
        conv2, bn2 = seq[3], seq[4]

        dc_conv1, dc_bn1, _, dc_conv2, dc_bn2, _ = dc_block.blocks[i]

        dc_conv1.weight.data.copy_(conv1.weight.data)
        dc_conv1.bias.data.copy_(conv1.bias.data)
        copy_bn(bn1, dc_bn1)

        dc_conv2.weight.data.copy_(conv2.weight.data)
        dc_conv2.bias.data.copy_(conv2.bias.data)
        copy_bn(bn2, dc_bn2)


def copy_scaleprediction(dense_pred, dc_pred):
    conv1, bn1 = dense_pred.pred[0], dense_pred.pred[1]
    dc_pred.conv1.weight.data.copy_(conv1.weight.data)
    dc_pred.conv1.bias.data.copy_(conv1.bias.data)
    copy_bn(bn1, dc_pred.bn1)

    conv2 = dense_pred.pred[3]  # plain conv, no BN
    dc_pred.conv2.weight.data.copy_(conv2.weight.data)
    if conv2.bias is not None:
        dc_pred.conv2.bias.data.copy_(conv2.bias.data)


def load_trained_weights_into_dc(dense_model, dc_model):
    dl = dense_model.layers  # indices 0-29, in build order

    cnnblock_map = [
        (dl[0],  dc_model.b1),  (dl[1],  dc_model.b2),  (dl[3],  dc_model.b3),
        (dl[5],  dc_model.b4),  (dl[7],  dc_model.b5),  (dl[9],  dc_model.b6),
        (dl[11], dc_model.b7),  (dl[12], dc_model.b8),  (dl[14], dc_model.b9),
        (dl[16], dc_model.b10), (dl[18], dc_model.b11), (dl[19], dc_model.b12),
        (dl[21], dc_model.b13), (dl[23], dc_model.b14), (dl[25], dc_model.b15),
        (dl[26], dc_model.b16), (dl[28], dc_model.b17),
    ]
    for dense_block, dc_block in cnnblock_map:
        copy_cnnblock(dense_block, dc_block)

    residual_map = [
        (dl[2],  dc_model.r1), (dl[4],  dc_model.r2), (dl[6],  dc_model.r3),
        (dl[8],  dc_model.r4), (dl[10], dc_model.r5), (dl[13], dc_model.r6),
        (dl[20], dc_model.r7), (dl[27], dc_model.r8),
    ]
    for dense_block, dc_block in residual_map:
        copy_residualblock(dense_block, dc_block)

    scale_map = [
        (dl[15], dc_model.pred1), (dl[22], dc_model.pred2), (dl[29], dc_model.pred3),
    ]
    for dense_pred, dc_pred in scale_map:
        copy_scaleprediction(dense_pred, dc_pred)

    print("Weight transfer complete.")


# --------------------------------------------------------------------------
# Build, load, transfer, and sanity-check
# --------------------------------------------------------------------------

if __name__ == "__main__":
    leanring_rate = 1e-4
    checkpoint_file = "checkpoint.pth.tar"

    # 1. Dense model — note: moved with memory_format=channels_last too,
    #    matching the official reference example.
    model = YOLOv3(num_classes=20).to(device, memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=leanring_rate)
    load_checkpoint(checkpoint_file, model, optimizer, leanring_rate, device)
    model.eval()

    # 2. DC model — also channels_last, per reference example
    dc_model = DCYOLOv3(num_classes=20).to(device, memory_format=torch.channels_last)
    dc_model.eval()
    load_trained_weights_into_dc(model, dc_model)
    dc_model.process_filters()

    # 3. Sanity check: compare on a random static frame
    x = torch.randn(1, 3, 416, 416).to(device)
    x = x.contiguous(memory_format=torch.channels_last)

    with torch.no_grad():
        dense_out = model(x)
        dc_out = dc_model(x)

    for i in range(3):
        diff = (dense_out[i] - dc_out[i]).abs().max()
        print(f"Scale {i} max diff: {diff.item()}")
