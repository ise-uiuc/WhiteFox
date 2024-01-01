
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 24, 1, stride=2, padding=1)
    def forward(self, x1, other=True, x6=False):
        v1 = self.conv(x1)
        if other == True:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        if x6 == False:
            x6 = torch.randn(v2.shape)
        v4 = v1 - x6
        return v4
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
