
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(3, 7, 1, stride=1, padding=0)
        self.conv1b = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(9, 2, 5, stride=1, padding=2)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = self.conv1b(x1)
        v2a = torch.relu(v1a)
        v2b = torch.relu(v1b)
        v3 = torch.cat([v2a, v2b], 1)
        v4 = self.conv2(v3)
        v5 = torch.max_pool2d(v4, 2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
