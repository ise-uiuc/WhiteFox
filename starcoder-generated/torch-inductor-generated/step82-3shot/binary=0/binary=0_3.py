
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(100, 32, 3, stride=1, padding=1)
    def forward(self, x1, other=None, padding1=None, padding2=True, padding3=torch.rand(32, 288, 1, 1)):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 100, 288, 3)
