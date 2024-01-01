
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, 3, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(96, 512, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = self.conv1(x1)
        v6 = self.conv1(x1)
        v7 = self.conv2(x1)
        v9 = self.conv2(x1)
        v10 = self.conv2(x1)
        v11 = v9 + v7
        v12 = v10 + v3
        v13 = v11 + v12
        v14 = v13 + v5
        v15 = v14 + v6
        v16 = v15 + v4
        v17 = v16 + v1
        v18 = torch.relu(v17)
        return v18
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
