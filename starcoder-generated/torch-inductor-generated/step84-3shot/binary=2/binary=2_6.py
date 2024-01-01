
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 12, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - (-0.0341)
        return v2
# Inputs to the model
x2 = torch.randn(1, 10, 35, 35)
