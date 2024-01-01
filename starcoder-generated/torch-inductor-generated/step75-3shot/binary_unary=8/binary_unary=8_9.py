
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1
        v3 = v1
        v4 = self.conv1(x1)
        v5 = self.conv1(x1)
        v6 = v2 + v3 + v4 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
