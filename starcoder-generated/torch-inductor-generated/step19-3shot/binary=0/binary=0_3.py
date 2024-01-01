
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 1, stride=1, padding=4)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other is None:
            v1 = v1 + x1
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
