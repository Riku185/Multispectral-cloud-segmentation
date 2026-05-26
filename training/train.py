import os
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets.dataset_loader import CloudDataset
from models.mobilenet_unet import MobileNetUNet

import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def dice_loss(pred, target, smooth=1):

    pred = torch.sigmoid(pred)

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    dice = (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


def train():

    config = yaml.safe_load(open("configs/config.yaml"))

    img_dir = config["dataset"]["image_dir"]
    mask_dir = config["dataset"]["mask_dir"]

    images = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.endswith(".tif")
    ])

    masks = [
        os.path.join(mask_dir, os.path.basename(f).replace("TOA", "CLD"))
        for f in images
    ]

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images,
        masks,
        test_size=config["training"]["validation_split"],
        random_state=42
    )

    train_ds = CloudDataset(train_imgs, train_masks, augment=True)
    val_ds = CloudDataset(val_imgs, val_masks, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,          # better for Windows
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        num_workers=2,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = MobileNetUNet().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    scaler = GradScaler()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["max_epochs"]):

        print("\nEpoch:", epoch + 1)

        model.train()
        train_loss = 0

        for img, mask in tqdm(train_loader):

            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():

                pred = model(img)

                pred = F.interpolate(
                    pred,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                loss = loss_fn(pred, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for img, mask in val_loader:

                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with autocast():

                    pred = model(img)

                    pred = F.interpolate(
                        pred,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                    loss = loss_fn(pred, mask)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")


        if val_loss < best_loss:

            best_loss = val_loss
            torch.save(model.state_dict(), "best_cloud_model.pth")

            print("✔ Best model saved")

            patience_counter = 0

        else:

            patience_counter += 1
            print("No improvement. Patience:", patience_counter)


        if patience_counter >= config["training"]["patience"]:

            print("\nEarly stopping triggered.")
            break