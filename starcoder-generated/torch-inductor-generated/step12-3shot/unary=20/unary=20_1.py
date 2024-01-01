
import torch.nn as nn
import torch.nn.functional as F
class Unet_block_1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = Unet_block_1(1, 32)
        self.block_2 = Unet_block_1(in_c=32, out_c=64)
        self.block_3 = Unet_block_1(in_c=64, out_c=128)
        self.block_4 = Unet_block_1(in_c=128, out_c=128)
        self.conv_t1 = nn.ConvTranspose2d(128, 64, 3, 2, 1)
        self.block_4_2 = Unet_block_1(in_c=128, out_c=64)
        self.conv_t2 = nn.ConvTranspose2d(64, 32, 3, 2, 1)
        self.block_5 = Unet_block_1(in_c=64, out_c=32)
        self.conv_t3 = nn.ConvTranspose2d(32, 1, 3, 2, 1)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = F.relu(self.conv_t1(x))
        x = torch.cat((torch.sigmoid(self.conv_t1(x)), x), 1)
        x = self.block_4_2(x)
        x = F.relu(self.conv_t2(x))
        x = torch.cat((torch.sigmoid(self.conv_t2(x)), x), 1)
        x = self.block_5(x)
        x = F.relu(self.conv_t3(x))
        x = torch.sigmoid(self.conv_t3(x))
        return x

model = Model()
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
