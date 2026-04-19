import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CelebAHQDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)
