
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1.sum()
# Inputs to the model
x1 = torch.randn(1, 1, 8, 32)
