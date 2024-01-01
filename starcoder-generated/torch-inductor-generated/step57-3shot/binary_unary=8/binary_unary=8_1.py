
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv1(v2)
        v4 = self.conv2(v3)
        v5 = v4 * x1
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
