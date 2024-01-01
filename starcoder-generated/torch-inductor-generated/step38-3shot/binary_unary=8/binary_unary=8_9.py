
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = self.conv2(x1)
        v3 = self.conv2(x1)
        v4 = self.conv2(x1)
        v5 = self.conv2(x1)
        v6 = self.conv2(x1)
        v7 = v2 + v3 + v4 + v5 + v6
        v8 = self.conv2(x1)
        v9 = v1 + v7 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
