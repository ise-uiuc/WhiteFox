
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv(x1))
        v2 = torch.relu(self.conv(x1))
        v3 = torch.relu(self.conv(x1))
        v4 = torch.relu(v1 + v2 + v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
