
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        a1 = self.conv(x1)
        a2 = a1 * a1 + 2 * a1 + 11
        a3 = torch.nn.functional.hardtanh(a2, 0, 6)
        a4 = a1 * a3
        a5 = a4 / 6
        return a5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
