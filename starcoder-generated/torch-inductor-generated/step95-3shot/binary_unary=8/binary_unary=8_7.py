
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + self.conv1(x1)
        v3 = torch.relu(v2)
        return torch.relu(v3)
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
