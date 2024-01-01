
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 192, 1, stride=1, padding=0)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv(x1)
        v2 = v1 + 1.3
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
