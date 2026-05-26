import os
import torch
import numpy as np
import tifffile as tiff
import torch.nn.functional as F

from models.mobilenet_unet import MobileNetUNet

device = "cuda" if torch.cuda.is_available() else "cpu"

SELECTED_BANDS = [1,2,3,7]


def preprocess_mask(mask):
    """
    Convert dataset labels:
    0   -> fill value (ignored)
    64  -> cloud shadow
    128 -> clear
    255 -> cloud

    Into model classes:
    0 -> clear
    1 -> shadow
    2 -> cloud
    """

    new_mask = np.zeros_like(mask)

    new_mask[mask == 128] = 0
    new_mask[mask == 64] = 1
    new_mask[mask == 255] = 2

    return new_mask.astype(np.int64)


def predict(model, img):

    img = img[SELECTED_BANDS,:,:]
    img = img.astype(np.float32) / 10000.0

    input_tensor = torch.tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(input_tensor)

        pred = F.interpolate(
            pred,
            size=(512,512),
            mode="bilinear",
            align_corners=False
        )

        pred = torch.softmax(pred, dim=1)

    mask = torch.argmax(pred, dim=1)
    mask = mask.squeeze().cpu().numpy()

    return mask


def compute_metrics(pred, gt):

    # Ignore fill values
    valid_pixels = gt != 0

    pred = pred[valid_pixels]
    gt = gt[valid_pixels]

    if len(gt) == 0:
        return 0, 0

    # Pixel accuracy
    accuracy = np.sum(pred == gt) / len(gt)

    # IoU per class
    classes = [0,1,2]
    ious = []

    for c in classes:

        pred_c = pred == c
        gt_c = gt == c

        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()

        if union == 0:
            continue

        ious.append(intersection / union)

    mean_iou = np.mean(ious) if len(ious) > 0 else 0

    return accuracy, mean_iou


def evaluate(image_dir, mask_dir, model_path):

    model = MobileNetUNet()

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()

    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))

    accuracies = []
    ious = []

    for img_name, mask_name in zip(images, masks):

        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        img = tiff.imread(img_path)

        gt = tiff.imread(mask_path)

        gt = preprocess_mask(gt)

        pred = predict(model, img)

        # Debug check (optional)
        # print("GT unique:", np.unique(gt))
        # print("Pred unique:", np.unique(pred))

        acc, iou = compute_metrics(pred, gt)

        accuracies.append(acc)
        ious.append(iou)

    print("\nEvaluation Results")
    print("-------------------")
    print("Images evaluated:", len(accuracies))
    print("Mean Pixel Accuracy:", np.mean(accuracies))
    print("Mean IoU:", np.mean(ious))


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    image_dir = os.path.join(BASE_DIR, "data/raw/images")
    mask_dir = os.path.join(BASE_DIR, "data/raw/masks")

    model_path = os.path.join(BASE_DIR, "best_cloud_model.pth")

    evaluate(image_dir, mask_dir, model_path)