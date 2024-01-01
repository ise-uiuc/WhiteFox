
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(v1)
        v3 = torch.relu(v1 + v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v3 + v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
