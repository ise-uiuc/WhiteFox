
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 6), stride=1)
    def forward(self, x1):
        v1 = torch.clamp(self.conv(x1) + 3, 0, 6)
        v2 = v1 / 6
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
