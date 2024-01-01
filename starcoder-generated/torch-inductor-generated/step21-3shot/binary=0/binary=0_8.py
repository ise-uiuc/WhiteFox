
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 12, 3, stride=1, padding=1)
    def forward(self, x1, other, padding1=None, padding2=None, padding3=None, padding4=None, padding5=None):
        v1 = self.conv(x1)
        t1 = v1 + other
        return t1
# Inputs to the model
x1 = torch.randn(2, 50, 64, 64)
