
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        v1 = self.conv2(x1)
        v2 = t1 + v1
        v3 = torch.relu(v2)
        v4 = v2 + v3
        v5 = v1 + v4
        v6 = v5 + self.conv1(x1)
        v7 = v6 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
