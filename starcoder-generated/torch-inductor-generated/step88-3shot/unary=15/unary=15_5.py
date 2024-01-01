
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2a = torch.relu(v1)
        v2b = torch.relu(v1)
        v3 = self.conv2(v2a)
        v4a = torch.relu(v3)
        v4b = torch.relu(v3)
        return (v4a, v4a, v4a, v4a, v4a, v4a, v4b, v4b, v4b, v4b, v4b)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
