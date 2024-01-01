
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x5):
        v1 = self.conv(x5)
        v2 = v1 - float(0.0)
        return v2
# Inputs to the model
x5 = torch.randn(1, 1, 381, 511)
