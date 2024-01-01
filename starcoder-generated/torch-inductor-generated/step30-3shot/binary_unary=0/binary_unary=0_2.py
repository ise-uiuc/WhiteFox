
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.ones_like(v1)
        v3 = v2 + v1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
