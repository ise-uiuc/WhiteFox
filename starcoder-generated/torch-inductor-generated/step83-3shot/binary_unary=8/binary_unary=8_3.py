
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=1)
        
        self.pool = torch.nn.AvgPool2d(stride=2, kernel_size=2)
        self.adaptiveAvgPool = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x1):
        v1 = self.pool(self.conv1(x1))
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv2(v1)
        v5 = self.pool(self.conv2(v1))
        v6 = self.conv3(v5)
        v7 = self.pool(v5)
        v8 = self.conv3(v7)
        v9 = self.conv1(v1)
        v10 = self.adaptiveAvgPool(v9)
        v11 = self.conv1(v1)
        v12 = self.conv2(v9)
        v13 = self.conv3(v10)
        v14 = self.pool(v12)
        v15 = self.conv3(v14)
        v16 = self.pool(v11)
        v17 = v13 + v15 + v16
        v18 = torch.relu(v17)
        return v18
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
