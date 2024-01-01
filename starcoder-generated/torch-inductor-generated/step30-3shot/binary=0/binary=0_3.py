
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, wht=None):
        v1 = self.conv(x1)
        return v1 + wht
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
