
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        v5 = torch.relu(v3)
        v6 = torch.relu(v4)
        v7 = v5 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
