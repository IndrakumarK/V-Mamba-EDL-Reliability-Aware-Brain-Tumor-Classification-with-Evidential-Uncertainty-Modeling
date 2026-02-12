import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        # Convert grayscale to RGB (channel triplication)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
