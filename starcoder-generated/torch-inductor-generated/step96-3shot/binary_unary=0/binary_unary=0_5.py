
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.identity(v1)
        v3 = self.identity(x)
        v4 = v2 + v3
        v5 = v1 + v2
        v6 = torch.relu(v4)
        return v6
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
