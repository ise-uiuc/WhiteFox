
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(57, 64, 1, stride=1, padding=0)
        self.add = torch.nn.add
    def forward(self, x1, other=0, padding1=None):
        v1 = self.conv(x1)
        v2 = self.add(v1, other)
        return v2
# Inputs to the model
x1 = torch.randn(3, 57, 64, 64)
