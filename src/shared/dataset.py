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
        img = Image.open(self.paths[idx]).convert('RGB
        return self.transform(img)

class FFHQDataset(Dataset): 
    HF_DATASET = "bitmind/ffhq-256"
 
    def __init__(self, data_dir, image_size=256, download=False, max_images=None):
        self.data_dir   = data_dir
        self.image_size = image_size
 
        if download:
            self._download()
 
        self.paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(('.jpg', '.png'))
        ])
 
        if max_images is not None:
            self.paths = self.paths[:max_images]
 
        if len(self.paths) == 0:
            raise RuntimeError(
                f"No images found in '{data_dir}'. "
                "Run with download=True, or place FFHQ images there manually."
            )
 
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
 
    def _download(self):
        """Download FFHQ thumbnails from Hugging Face and save as individual images."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install the Hugging Face datasets library:\n"
                "    pip install datasets"
            )
 
        os.makedirs(self.data_dir, exist_ok=True)
 
        # Check if already downloaded
        existing = [f for f in os.listdir(self.data_dir) if f.endswith(('.jpg', '.png'))]
        if len(existing) > 0:
            print(f"Found {len(existing)} existing images in '{self.data_dir}', skipping download.")
            return
 
        print(f"Downloading FFHQ thumbnails from Hugging Face ({self.HF_DATASET})...")
        hf_dataset = load_dataset(self.HF_DATASET, split='train')
 
        for i, sample in enumerate(hf_dataset):
            img = sample['image']          # PIL Image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            save_path = os.path.join(self.data_dir, f"{i:05d}.png")
            img.save(save_path)
            if (i + 1) % 5000 == 0:
                print(f"  Saved {i+1}/{len(hf_dataset)} images...", flush=True)
 
        print(f"Download complete. {len(hf_dataset)} images saved to '{self.data_dir}'.")
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)
 