from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class HumanDataset(Dataset):
    def __init__(self, root_dir,limit=None):
        self.root_dir = Path(root_dir)
        self.samples = []
        
        for label in ['0', '1']:
            for img_path in (self.root_dir / label).glob('*'):
                self.samples.append((img_path, int(label)))
        if limit is not None:
          self.samples = self.samples[:limit]


        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((96, 96)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        bbox = torch.zeros(4)

        return image, torch.tensor(label, dtype=torch.long), bbox


