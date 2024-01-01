
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 8, 1, stride=1, padding=1)
    def forward(self, x1, x0=None):
        v1 = self.conv(x1)
        if x0 == None:
            x0 = torch.randn(v1.shape)
        v2 = v1 + x0
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
