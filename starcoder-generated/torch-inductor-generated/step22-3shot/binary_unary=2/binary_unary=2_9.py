
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 6, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 10, 6, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = torch.flatten(v3)
        v5 = self.conv2(v4)
        v6 = v5 - 0.6
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
