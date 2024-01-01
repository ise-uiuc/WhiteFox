
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(249, 21, 1, stride=1, padding=0)
        self.conv1b = torch.nn.Conv2d(249, 21, 1, stride=1, padding=0)
        self.conv2a = torch.nn.Conv2d(21, 124, 1, stride=1, padding=0)
        self.conv2b = torch.nn.Conv2d(21, 124, 1, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(124, 10)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = self.conv1b(x1)
        v2a = torch.relu(v1a)
        v2b = torch.relu(v1b)
        v3a = self.conv2a(v2a)
        v3b = self.conv2b(v2b)
        v4ab = torch.add(v3a, v3b)
        v4 = torch.relu()
        v5 = self.linear1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 249, 32, 32)
