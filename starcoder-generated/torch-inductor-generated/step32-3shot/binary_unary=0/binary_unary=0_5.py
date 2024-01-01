
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        return torch.abs(v5)
# Inputs to the model
x = torch.randn(2, 16, 64, 64)
