import torch.nn as nn

class CircleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # input images 1 * 96 * 96
        self.L1_conv = nn.Sequential(nn.Conv2d(1, 32, 5),    nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2, 2)) # 32 * 46 * 46
        self.L2_conv = nn.Sequential(nn.Conv2d(32, 64, 3),   nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2, 2)) # 64 * 22 * 22
        self.L3_conv = nn.Sequential(nn.Conv2d(64, 128, 3),  nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)) # 128 * 10 * 10
        self.L4_conv = nn.Sequential(nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)) # 256 * 4 * 4
        self.L5_conv = nn.Sequential(nn.Conv2d(256, 512, 3), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2)) # 512 * 1 * 1
        self.L6_fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU()) # 256
        self.L7_fc = nn.Sequential(nn.Linear(256, 2)) # 2
        
    def forward(self, x):
        x = self.L1_conv(x)
        x = self.L2_conv(x)
        x = self.L3_conv(x)
        x = self.L4_conv(x)
        x = self.L5_conv(x)
        x = x.view(-1, 512)
        x = self.L6_fc(x)
        x = self.L7_fc(x)
        return x