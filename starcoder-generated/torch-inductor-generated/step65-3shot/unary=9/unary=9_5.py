
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = nn.functional.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 + 3
        v5 = torch.clamp(v4, 0, 6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
