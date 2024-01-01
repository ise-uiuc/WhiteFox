
class PatternModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3)
    def forward(self, x1, x2):
        v1 = self.conv(x2)
        v2 = x1 * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 128, 128)
