
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1)
        self.conv1 = torch.nn.Conv2d(64, 16, 3, stride=1)
    def forward(self, X0):
        v1 = self.conv(X0)
        v2 = self.conv1(v1)
        v3 = v2 - 32
        v4 = F.relu(v3)
        return v4
# Inputs to the model
X0 = torch.randn(1, 3, 64, 64)
