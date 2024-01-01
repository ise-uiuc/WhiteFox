
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 7, 4, stride=1, padding=3)
    def forward(self, x1, other=None, padding1=None, padding2=False, padding3=True):
        v1 = self.conv(x1)
        if other == None:
            other = torch.zeros(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 124, 124)
