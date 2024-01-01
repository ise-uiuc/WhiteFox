
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 17, 1, stride=1, padding=1)
    def forward(self, x1, alpha=0.0, beta=-1.0):
        v1 = self.conv(x1)
        v2 = v1 + alpha
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 64, 64)
