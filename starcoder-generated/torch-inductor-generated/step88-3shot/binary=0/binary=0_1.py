
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 8, 9, stride=2, padding=3)
    def forward(self, x1, other=None, padding1=None, padding2=None):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 20, 20)
