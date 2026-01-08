import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class HumanDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=96):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.Grayscale(num_output_channels=1),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor()
        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Load labels
        label_path = os.path.join(
            self.labels_dir,
            image_name.replace(".jpg", ".txt")
                         .replace(".png", ".txt")
        )

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    _, x, y, w, h = parts
                    boxes.append([x, y, w, h])
        MAX_HUMANS = 6
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if len(boxes) > MAX_HUMANS:
            areas = boxes[:, 2] * boxes[:, 3]  # w * h
            topk = torch.argsort(areas, descending=True)[:MAX_HUMANS]
            boxes = boxes[topk]

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            


        return image, boxes
