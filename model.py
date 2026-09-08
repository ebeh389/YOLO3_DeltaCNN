"""
model.py — reusable YOLOv3 code: model classes, loss, dataset, and utility
functions. Safe to import: no dataset loading, plotting, or training runs
at import time. Both train.py and model_deltacnn.py import from this file.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------

def iou(box1, box2, is_pred=True):
    """Intersection over Union for two sets of boxes."""
    if is_pred:
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        epsilon = 1e-6
        return intersection / (union + epsilon)

    else:
        # IoU based on width/height only (anchor matching)
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                             torch.min(box1[..., 1], box2[..., 1])
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area

#old version of nms
'''
def nms(bboxes, iou_threshold, threshold):
    """Non-maximum suppression to remove overlapping bounding boxes."""
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_nms = []

    while bboxes:
        first_box = bboxes.pop(0)
        for box in bboxes:
            if box[0] != first_box[0] or iou(
                torch.tensor(first_box[2:]),
                torch.tensor(box[2:]),
            ) < iou_threshold:
                if box not in bboxes_nms:
                    bboxes_nms.append(box)

    return bboxes_nms
'''
def nms(bboxes, iou_threshold, threshold):
    """Non-maximum suppression to remove overlapping bounding boxes."""
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)   # always keep the current best box

        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            ) < iou_threshold
        ]

    return bboxes_after_nms


def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    """Convert grid-cell-relative predictions/targets into bounding boxes."""
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )

    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]

    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)

    return converted_bboxes.tolist()


def plot_image(image, boxes, class_labels):
    """Plot an image with its bounding boxes and class labels.
    class_labels is passed in explicitly (not a module-level global)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    colour_map = plt.get_cmap("tab20b")
    colors = [colour_map(i) for i in np.linspace(0, 1, len(class_labels))]

    img = np.array(image)
    h, w, _ = img.shape

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * w, upper_left_y * h),
            box[2] * w,
            box[3] * h,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        ax.add_patch(rect)
        plt.text(
            upper_left_x * w,
            upper_left_y * h,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """device is now an explicit argument rather than a module-level global,
    since this file no longer defines `device` itself."""
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, image_dir, label_dir, anchors,
        image_size=416, grid_sizes=[13, 26, 52],
        num_classes=20, transform=None
    ):
        self.label_list = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.num_classes = num_classes
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.label_list.iloc[idx, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

        img_path = os.path.join(self.image_dir, self.label_list.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augs = self.transform(image=image, bboxes=bboxes)
            image = augs["image"]
            bboxes = augs["bboxes"]

        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) for s in self.grid_sizes]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, is_pred=False)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                s = self.grid_sizes[scale_idx]
                i, j = int(s * y), int(s * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = s * x - j, s * y - i
                    width_cell, height_cell = (width * s, height * s)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.01)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.01),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.01)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x)
            if self.use_residual:
                x = x + residual
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(2 * in_channels, (num_classes + 5) * 3, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ResidualBlock(1024, use_residual=False, num_repeats=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ScalePrediction(512, num_classes=num_classes),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ResidualBlock(512, use_residual=False, num_repeats=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ScalePrediction(256, num_classes=num_classes),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2),
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256, use_residual=False, num_repeats=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            ScalePrediction(128, num_classes=num_classes)
        ])

    def forward(self, x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, anchors):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]), (target[..., 0:1][no_obj]),
        )

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([
            self.sigmoid(pred[..., 1:3]),
            torch.exp(pred[..., 3:5].clamp(max=10)) * anchors
        ], dim=-1)
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])

        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long())

        return box_loss + object_loss + no_object_loss + class_loss
