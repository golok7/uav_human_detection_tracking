import torch
import cv2
import os
import numpy as np
from torchvision.ops import nms

from data.models.cnn_model import HumanDetectionCNN
from training.detection_dataset import HumanDetectionDataset

# -------------------------------
# YOLO → pixel box
# -------------------------------
def yolo_to_pixel(box, img_w, img_h):
    xc, yc, w, h = box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]

# -------------------------------
# Main visualization
# -------------------------------
def visualize_predictions(
    num_images=5,
    conf_thresh=0.4,
    iou_thresh=0.4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HumanDetectionCNN().to(device)
    model.load_state_dict(
        torch.load("data/models/human_detector.pth", map_location=device)
    )
    model.eval()

    dataset = HumanDetectionDataset(
        images_dir="dataset/valid/images",
        labels_dir="dataset/valid/labels"
    )

    os.makedirs("outputs", exist_ok=True)

    for idx in range(num_images):
        image, gt_boxes = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_boxes, pred_confs = model(img_tensor)

        pred_boxes = pred_boxes[0].cpu()   # [N, 4]
        pred_confs = pred_confs[0].cpu()   # [N]

        # ---------------------------
        # Confidence filtering
        # ---------------------------
        keep = pred_confs > conf_thresh
        pred_boxes = pred_boxes[keep]
        pred_confs = pred_confs[keep]

        if len(pred_boxes) == 0:
            continue

        # ---------------------------
        # Convert to pixel boxes
        # ---------------------------
        img = image.squeeze().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        pixel_boxes = torch.tensor(
            [yolo_to_pixel(b, w, h) for b in pred_boxes],
            dtype=torch.float32
        )

        # ---------------------------
        # NMS
        # ---------------------------
        keep_idx = nms(pixel_boxes, pred_confs, iou_thresh)
        pixel_boxes = pixel_boxes[keep_idx]

        # ---------------------------
        # Draw GT (GREEN)
        # ---------------------------
        for box in gt_boxes:
            x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

        # ---------------------------
        # Draw Predictions (RED)
        # ---------------------------
        for box in pixel_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        out_path = f"outputs/result_{idx}.png"
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize_predictions(num_images=5)
