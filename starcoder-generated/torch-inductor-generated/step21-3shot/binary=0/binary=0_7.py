
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(42, 75, 1, stride=1, padding=1)
    def forward(self, x1, other=None, padding1=None, other1=True, padding2=None, other2=True, padding3=None, other3=True, padding4=None, other4=True):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 42, 64, 64)
