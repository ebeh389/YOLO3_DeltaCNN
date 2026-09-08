import torch
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import matplotlib
matplotlib.use("TkAgg")

from tqdm import tqdm

from model import (
    YOLOv3, YOLOLoss, Dataset,
    iou, nms, convert_cells_to_bboxes, plot_image,
    save_checkpoint, load_checkpoint,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device is {device}')

# Load and save model variable
load_model = False
save_model = True

# model checkpoint file name
checkpoint_file = "checkpoint.pth.tar"

# Anchor boxes for each feature map scaled between 0 and 1
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

batch_size = 20 #32
leanring_rate = 1e-4
epochs = 70  #20
image_size = 416
s = [image_size // 32, image_size // 16, image_size // 8]
print(f'batch size = {batch_size}   learning rate = {leanring_rate}    epochs = {epochs}')
class_labels = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------

train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)

test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)


# --------------------------------------------------------------------------
# Training loop — plain fp32, gradient clipping (no autocast/GradScaler,
# per earlier NaN debugging: fp16 overflowed on this untrained deep network)
# --------------------------------------------------------------------------

def training_loop(loader, model, optimizer, loss_fn, scaled_anchors):
    progress_bar = tqdm(loader, leave=True)
    losses = []

    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)

        outputs = model(x)
        loss = (
            loss_fn(outputs[0], y0, scaled_anchors[0])
            + loss_fn(outputs[1], y1, scaled_anchors[1])
            + loss_fn(outputs[2], y2, scaled_anchors[2])
        )

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)


# --------------------------------------------------------------------------
# Preview a random training sample with its ground-truth boxes drawn on it
# --------------------------------------------------------------------------

def preview_random_sample():
    dataset = Dataset(
        csv_file="./dataset/train.csv",
        image_dir="./dataset/train/images",
        label_dir="./dataset/train/labels",
        grid_sizes=[13, 26, 52],
        anchors=ANCHORS,
        transform=test_transform
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )

    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )

    x, y = next(iter(loader))

    boxes = []
    for i in range(y[0].shape[1]):
        anchor = scaled_anchors[i]
        boxes += convert_cells_to_bboxes(
                    y[i], is_predictions=False, s=y[i].shape[2], anchors=anchor
                )[0]

    boxes = nms(boxes, iou_threshold=1, threshold=0.7)
    plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes, class_labels)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def run_training():
    model = YOLOv3(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=leanring_rate)
    loss_fn = YOLOLoss()

    if load_model:
        load_checkpoint(checkpoint_file, model, optimizer, leanring_rate, device)

    train_dataset = Dataset(
        csv_file="./dataset/train.csv",
        image_dir="./dataset/train/images/",
        label_dir="./dataset/train/labels/",
        anchors=ANCHORS,
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    scaled_anchors = (
        torch.tensor(ANCHORS) *
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    for e in range(1, epochs + 1):
        print("Epoch:", e)
        training_loop(train_loader, model, optimizer, loss_fn, scaled_anchors)

        if save_model:
            save_checkpoint(model, optimizer, filename=checkpoint_file)


# --------------------------------------------------------------------------
# Inference on a sample image
# --------------------------------------------------------------------------

def run_inference_sample():
    model = YOLOv3(num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=leanring_rate)
    load_checkpoint(checkpoint_file, model, optimizer, leanring_rate, device)

    test_dataset = Dataset(
        csv_file="./dataset/valid.csv",
        image_dir="./dataset/valid/images/",
        label_dir="./dataset/valid/labels/",
        anchors=ANCHORS,
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=True,
    )

    x, y = next(iter(test_loader))
    x = x.to(device)

    anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        bboxes = [[] for _ in range(x.shape[0])]

        for i in range(3):
            obj = torch.sigmoid(output[i][..., 0])
            print(f"Scale {i}:")
            print("Min objectness:", obj.min().item())
            print("Max objectness:", obj.max().item())
            batch_size_, A_, S, _, _ = output[i].shape
            anchor = anchors[i]
            boxes_scale_i = convert_cells_to_bboxes(output[i], anchor, s=S, is_predictions=True)
            for idx, box in enumerate(boxes_scale_i):
                bboxes[idx] += box
    model.train()

    for i in range(x.shape[0]):
        nms_boxes = nms(bboxes[i], iou_threshold=0.35, threshold=0.75)     #iou 0.5   threshld=0.6
        print("Number of predictions:", len(nms_boxes))
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes, class_labels)


if __name__ == "__main__":
    preview_random_sample()    # shows one random training image with its ground-truth boxes
    #run_training()
    run_inference_sample()
