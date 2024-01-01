
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(1, 3, 3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv1c = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = torch.relu(v1a)
        v2a = self.conv1b(v1b)
        v2b = torch.relu(v2a)
        v3a = self.conv1c(v2b)
        v3b = torch.relu(v3a)
        return v3b
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
