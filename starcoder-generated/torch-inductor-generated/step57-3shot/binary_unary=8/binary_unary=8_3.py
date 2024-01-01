
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv1(x1)
        v4 = self.conv2(v3)
        v5 = self.conv1(x1)
        v6 = self.conv2(v5)
        v7 = torch.add(v2, v4)
        v8 = torch.add(v6, v7)
        v9 = torch.relu(v8)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
