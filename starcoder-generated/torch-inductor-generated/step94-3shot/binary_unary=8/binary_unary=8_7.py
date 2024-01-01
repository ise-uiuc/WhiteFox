
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 8, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.relu(v1)
        v2 = self.conv2(v1)
        v2 = torch.relu(v2)
        v3 = self.conv2(v2)
        v3 = torch.relu(v3)
        v4 = v1 + v2 + v3
        v4 = torch.relu(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
