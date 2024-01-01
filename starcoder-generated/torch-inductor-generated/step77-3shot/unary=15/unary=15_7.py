
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(169, 32, 1, stride=1, padding=0)
        self.conv1b = torch.nn.Conv2d(169, 32, 1, stride=1, padding=0)
        self.conv2a = torch.nn.Conv2d(32, 100, 1, stride=1, padding=0)
        self.conv2b = torch.nn.Conv2d(32, 100, 1, stride=1, padding=0)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = self.conv1b(x1)
        v2a = torch.relu(v1a)
        v2b = torch.relu(v1b)
        v3a = self.conv2a(v2a)
        v3b = self.conv2b(v2b)
        return (v3a, v3b)
# Inputs to the model
x1 = torch.randn(1, 169, 32, 32)
