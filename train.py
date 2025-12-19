import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import argparse
from tqdm import tqdm
from src.model import EcoLineNet

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)[:, 1]
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        return 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()

class LineDataset(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(self.img_dir, name)), (320, 320)), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(cv2.imread(os.path.join(self.mask_dir, name.replace('.jpg', '.png')), 0), (320, 320), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(img).permute(2,0,1).float()/255.0, (torch.from_numpy(mask).float()/255.0 > 0.5).float()

def train(color_mode, epochs):
    data_path = f"data/{color_mode}_real"
    save_path = f"models/nfld_{color_mode}.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔥 학습 시작: {color_mode} (Dice Loss)")
    dataset = LineDataset(data_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = EcoLineNet(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = DiceLoss()

    for epoch in range(epochs):
        model.train()
        loop = tqdm(loader, desc=f"Ep {epoch+1}/{epochs}")
        for imgs, masks in loop:
            optimizer.zero_grad()
            loss = criterion(model(imgs.to(device)), masks.to(device))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"🎉 모델 저장 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    train(args.color, args.epochs)