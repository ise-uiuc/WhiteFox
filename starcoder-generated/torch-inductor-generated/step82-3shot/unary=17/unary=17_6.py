
from torch.nn.modules.upsampling import Upsample
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(64, affine=False)
        self.conv1 = torch.nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.conv1_1 = torch.nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.conv1_2 = torch.nn.ConvTranspose2d(16, 3, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(3, affine=True)
        self.upsample = Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv1_1(v4)
        v6 = torch.relu(v5)
        v7 = self.conv1_2(v6)
        v8 = self.bn2(v7)
        v9 = self.upsample(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 32, 32)
