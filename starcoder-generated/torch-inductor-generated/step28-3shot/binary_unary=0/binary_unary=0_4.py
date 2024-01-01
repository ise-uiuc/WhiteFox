
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        with torch.no_grad():
            v3 = self.conv1(x3)
        v4 = torch.mean(v3, dim=(-1, -2))
        v5 = torch.mean(v3, dim=(-2, -1))
        v6 = v3.view(x1.shape[0], 1, -1)
        v7 = self.conv2(v6)
        v8 = v7 + v1
        v9 = self.conv3(v8)
        v10 = self.conv3(v9)
        v11 = v10 + v4
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(2, 16, 64, 64)
