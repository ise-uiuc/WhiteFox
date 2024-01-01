
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * 0.001
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv2(x)
        v6 = v5 > 0
        x = v6 * 0.1
        v7 = torch.where(v4, v4, v3)
        v8 = self.conv3(v7)
        v9 = v8 > 0
        v10 = v8 * 0.1
        v11 = torch.where(v9, v8, v10)
        return v11
# Inputs to the model
x1 = torch.randn(2, 8, 64, 64)
