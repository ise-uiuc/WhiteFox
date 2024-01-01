
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.rand()
        v3 = v1 * (v2 - 0.5)
        v4 = self.conv2(v3)
        v5 = v2 - 0.5
        v6 = torch.relu(v5)
        v7 = torch.max(v4, v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
