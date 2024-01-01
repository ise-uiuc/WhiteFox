
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.nn.functional.avg_pool2d(x1, 3, 3, 1, 1, 1)
        v2 = self.conv(v1)
        v3 = v2 - 1300
        v4 = F.relu(v3)
        v5 = torch.nn.functional.avg_pool2d(x1, 3, 3, 1, 1, 1)
        v6 = self.conv(v5)
        v7 = v6 - 1300
        v8 = F.relu(v7)
        v9 = v8 + v4
        return v9
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
