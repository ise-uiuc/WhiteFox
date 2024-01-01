
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x1):
        y = self.conv(x1)
        y = F.hardtanh(y)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
