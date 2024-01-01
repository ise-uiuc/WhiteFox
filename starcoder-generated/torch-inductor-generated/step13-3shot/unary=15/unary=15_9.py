
# Example of multiple inputs
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, (5, 5), stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(x2)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        return v3, v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
x2 = torch.randn(1, 3, 256, 256)
x3 = torch.randn(2, 3, 256, 256)
x4 = torch.randn(1, 2, 256, 256)
