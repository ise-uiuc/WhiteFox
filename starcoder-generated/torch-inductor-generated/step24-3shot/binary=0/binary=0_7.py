
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0, groups=2)
    def forward(self, x1, x2=1):
        if x2 == None:
            x2 = torch.randn(x1.shape)
        v1 = self.conv(x1)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
