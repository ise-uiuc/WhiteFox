
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 2, stride=2)
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = v1 - 0.000653
        return v2
# Inputs to the model
x0 = torch.randn(2, 1, 16, 16)
