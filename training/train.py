from data.models.cnn_model import HumanDetectionCNN
from data.models.loss_functions import DetectionLoss
from training.detection_dataset import HumanDetectionDataset
from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HumanDetectionDataset(
    images_dir="dataset/train/images",  
    labels_dir="dataset/train/labels"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: list(zip(*x)))

model = HumanDetectionCNN(max_humans=6).to(device)
criterion = DetectionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = 0
    for images, gt_boxes in loader:
        images = torch.stack(images).to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]

        pred_boxes, pred_confs = model(images)
        loss = criterion(pred_boxes, pred_confs, gt_boxes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")

torch.save(
    model.state_dict(),
    "data/models/human_detector.pth"
)
print("✅ Model saved to data/models/human_detector.pth")
