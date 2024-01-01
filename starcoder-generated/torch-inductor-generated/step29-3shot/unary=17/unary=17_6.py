
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(6, 6, 1, padding=1, stride=2)
        self.conv1 = torch.nn.Conv2d(6, 32, 1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(64, 64, 1)
    def forward(self, x1):
        v6 = self.conv(x1)
        v7 = torch.sigmoid(v6)
        v8 = nn.functional.interpolate(v7, scale_factor=2, mode='nearest')
        v9 = self.conv1(v8)
        v10 = torch.sigmoid(v9)
        v11 = nn.functional.interpolate(v10, scale_factor=2, mode='nearest')
        v12 = self.conv2(v11)
        v13 = torch.sigmoid(v12)
        v14 = nn.functional.interpolate(v13, scale_factor=2, mode='nearest')
        v15 = self.conv3(v14)
        v16 = torch.sigmoid(v15)
        v17 = nn.functional.interpolate(v16, scale_factor=2, mode='nearest')
        v18 = self.conv4(v17)
        v19 = torch.sigmoid(v18)
        v20 = nn.functional.interpolate(v19, scale_factor=2, mode='nearest')
        v21 = self.conv5(v20)
        v22 = torch.sigmoid(v21)
        return v22
# Inputs to the model
x1 = torch.randn(1, 6, 242, 242)
