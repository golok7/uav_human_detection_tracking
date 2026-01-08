import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanDetectionCNN(nn.Module):
    def __init__(self, max_humans=6):
        super().__init__()
        self.max_humans = max_humans

        # Feature extractor (lightweight)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU()
        )

        # Detection heads
        self.bbox_head = nn.Linear(128, max_humans * 4)
        self.conf_head = nn.Linear(128, max_humans)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)

        boxes = torch.sigmoid(self.bbox_head(x)).view(-1, self.max_humans, 4)
        confs = torch.sigmoid(self.conf_head(x))

        return boxes, confs
