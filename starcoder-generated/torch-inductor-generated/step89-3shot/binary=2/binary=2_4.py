
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, 2, stride=2, padding=0)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - 1e-06
        return v2
# Inputs to the model
x3 = torch.randn(1, 4, 64, 64)
