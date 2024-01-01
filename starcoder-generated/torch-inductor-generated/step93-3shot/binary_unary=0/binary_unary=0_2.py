
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v3 = v1 + x2
        v2 = torch.relu(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
