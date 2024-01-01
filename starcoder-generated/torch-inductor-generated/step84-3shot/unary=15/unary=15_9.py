
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(1, 21, 3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(1, 21, 1, stride=1, padding=0)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = self.conv1b(x1)
        v2a = torch.relu(v1a)
        v2b = torch.relu(v1b)
        return v2a, v2b
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
