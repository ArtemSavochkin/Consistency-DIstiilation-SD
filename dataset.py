import os
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, path_to_dataset: str, filename: str = "valid_anno_repath.jsonl"):
        img_folders = [folder for folder in os.listdir(path_to_dataset) if folder != filename]
        self.data_dir = path_to_dataset
        with open(os.path.join(path_to_dataset, filename)) as f:
            self.data = [json.loads(data) for data in f.read().splitlines() if
                         json.loads(data)["img_path"].split("/")[1] in img_folders]
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Lambda(lambda x: 2 * x - 1)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        elem = self.data[idx]
        prompt = elem["prompt"]
        image = self.transform(Image.open(f"{self.data_dir}/{elem['img_path'][1:]}").convert("RGB"))
        return dict(prompt=prompt, image=image)


def get_dataloader(path_to_dataset, filename: str = "valid_anno_repath.jsonl", batch_size: int = 2):
    dataset = MyDataset(path_to_dataset, filename=filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
