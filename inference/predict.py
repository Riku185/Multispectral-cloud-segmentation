import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.mobilenet_unet import MobileNetUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

SELECTED_BANDS = [1, 2, 3, 7]


def predict(image_path, model_path=os.path.join(BASE_DIR, "best_cloud_model.pth")):

    model = MobileNetUNet()

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    img = tiff.imread(image_path)

    img = img[SELECTED_BANDS, :, :]

    img = img.astype(np.float32) / 10000.0

    input_tensor = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(input_tensor)

        pred = F.interpolate(
            pred,
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        )

        pred = torch.softmax(pred, dim=1)

    mask = torch.argmax(pred, dim=1)
    mask = mask.squeeze().cpu().numpy()

    return img, mask


def visualize_prediction(img, mask):

    # Create RGB satellite image
    rgb = np.stack([img[2], img[1], img[0]], axis=-1)
    rgb = np.clip(rgb * 3, 0, 1)

    # Create colored mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    # clear sky → black
    color_mask[mask == 0] = [0, 0, 0]

    # cloud shadow → blue
    color_mask[mask == 1] = [0, 0, 1]

    # cloud → white
    color_mask[mask == 2] = [1, 1, 1]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Satellite RGB")
    plt.imshow(rgb)

    plt.subplot(1, 3, 2)
    plt.title("Predicted Classes")
    plt.imshow(color_mask)

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(rgb)
    plt.imshow(color_mask, alpha=0.4)

    plt.show()


if __name__ == "__main__":

    image_path = os.path.join(
        BASE_DIR,
        "data/raw/images/200S2A_MSIL1C_20180102T084341_N0206_R064_T36TVP_20180102T110333.0256.0256.TOA.tif"
    )

    img, mask = predict(image_path)

    visualize_prediction(img, mask)