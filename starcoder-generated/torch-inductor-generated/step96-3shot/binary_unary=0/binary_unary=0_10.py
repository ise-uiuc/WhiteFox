
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.identity(x)
        v2 = self.identity(x)
        v3 = self.conv1(v1)
        v4 = v3 + v2
        v5 = self.conv1(v2)
        v6 = v5 + v3
        if x.shape == torch.Size([1, 16, 64, 64]):
            return v4 + v6
        else:
            v7 = torch.relu(v4)
            return v7
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
