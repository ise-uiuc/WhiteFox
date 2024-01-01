
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x1 = self.conv(x1)
        x2 = x1 + 3
        x3 = F.hardtanh(x2, min_val=0.0, max_val=6.0)
        x4 = x1 * x3
        x5 = x4 / 6
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
