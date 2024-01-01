
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1, other=5):
        v1 = self.conv(x1)
        v2 = other + v1
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
