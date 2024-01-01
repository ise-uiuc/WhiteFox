
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 24, 3, stride=1, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.6
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
