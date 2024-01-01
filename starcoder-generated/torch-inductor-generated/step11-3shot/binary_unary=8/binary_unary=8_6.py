
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
