
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = self.conv2(v3)
        v5 = torch.nn.functional.relu(v4)
        v6 = torch.clamp_max(v4, 6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
