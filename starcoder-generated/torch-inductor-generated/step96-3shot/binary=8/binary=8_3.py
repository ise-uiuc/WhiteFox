
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = torch.nn.functional.softplus(v1)
        v4 = torch.nn.functional.softplus(v2)
        v5 = v3.add(v4)
        v6 = self.bn1(x)
        v7 = self.bn2(x)
        v8 = v6 + v7
        v9 = v5 + v8
        return v9
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
