
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, bias=False, padding=1)
    def forward(self, x1, x2, other=10):
        v1 = self.conv(x1)
        v2 = x2 + v1
        return v2

# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
x2 = torch.randn(1, 3, 80, 80)
