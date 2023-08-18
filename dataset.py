import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for img_name in os.listdir(self.root_dir):
            img_path = os.path.join(self.root_dir, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(img_path)
                if image.mode == 'RGB':
                    self.data.append(img_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


dataset = CustomDataset(root_dir='/content', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)