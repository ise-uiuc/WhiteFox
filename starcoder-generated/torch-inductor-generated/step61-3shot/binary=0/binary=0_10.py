
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
    def forward(self, x1, padding1=None, padding2=None):
        v1 = self.conv(x1)
        v2 = v1 + padding1
        v3 = v2 + padding2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
