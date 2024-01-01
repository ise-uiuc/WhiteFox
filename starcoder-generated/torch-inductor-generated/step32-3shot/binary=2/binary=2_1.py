
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 1, stride=1, transposed=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
