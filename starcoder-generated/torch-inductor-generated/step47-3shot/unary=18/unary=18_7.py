
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv01 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # 32x64x64
        self.conv02 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1) # 32x32x32
        self.conv03 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # 64x16x16
        self.conv04 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1) # 64x8x8
        self.conv05 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=2, padding=1) # 64x1x1
        self.conv06 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8, stride=2, padding=1) # 64x1x1
        self.conv1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv01(x1) # 32x64x64 -> 32x32x32
        v2 = torch.sigmoid(v1)
        v3 = self.conv02(v2) # 32x32x32 -> 32x16x16
        v4 = torch.sigmoid(v3)
        v5 = self.conv03(v4) # 32x16x16 -> 64x8x8
        v6 = torch.sigmoid(v5)
        v7 = self.conv04(v6) # 64x8x8 -> 64x4x4
        v8 = torch.sigmoid(v7)
        v9 = self.conv05(v8) # 64x4x4 -> 128x2x2
        v10 = torch.sigmoid(v9)
        v11 = self.conv06(v10) # 128x2x2 -> 128x1x1
        v12 = torch.sigmoid(v11)
        v13 = self.conv1(v12) # 128x1x1 -> 64x1x1
        v14 = torch.sigmoid(v13)
        v15 = self.conv2(v14) # 64x1x1 -> 64x1x1
        v16 = torch.sigmoid(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
