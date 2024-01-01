
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.threshold(v1, 2.909919334294232e-07, 6.0)
        return v2
# Inputs to the model
x1 = torch.randn(8, 16, 19, 19)
