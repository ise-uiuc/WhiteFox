
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        return torch.clamp((2 * t1) + 3, 0, 6)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
