
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1)
    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = x1 + x1
        v1 = x2*x2
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
