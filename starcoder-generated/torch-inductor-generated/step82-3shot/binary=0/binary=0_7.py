
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 8, stride=2, padding=1, bias=False)
    def forward(self, x1, other=None, padding1=None, padding2=None, padding3=torch.ones(32)):
        v1 = self.conv(x1)
        if other == None:
            other = 1
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 20, 20, 20)
