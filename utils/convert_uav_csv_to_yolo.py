import csv
import os
from collections import defaultdict
from PIL import Image

CSV_PATH = "Drone_005-export.csv"
IMAGES_DIR = "uav_images"
OUTPUT_LABELS = "uav_labels"

os.makedirs(OUTPUT_LABELS, exist_ok=True)

# group boxes per image
image_boxes = defaultdict(set)

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_name = row["image"] + ".jpg"

        xmin = float(row["xmin"])
        ymin = float(row["ymin"])
        xmax = float(row["xmax"])
        ymax = float(row["ymax"])

        # avoid duplicate boxes (walk/push)
        image_boxes[img_name].add((xmin, ymin, xmax, ymax))

for img_name, boxes in image_boxes.items():
    img_path = os.path.join(IMAGES_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    w_img, h_img = Image.open(img_path).size

    label_path = os.path.join(
        OUTPUT_LABELS,
        img_name.replace(".jpg", ".txt")
    )

    with open(label_path, "w") as f:
        for xmin, ymin, xmax, ymax in boxes:
            xc = ((xmin + xmax) / 2) / w_img
            yc = ((ymin + ymax) / 2) / h_img
            bw = (xmax - xmin) / w_img
            bh = (ymax - ymin) / h_img

            f.write(f"0 {xc} {yc} {bw} {bh}\n")
