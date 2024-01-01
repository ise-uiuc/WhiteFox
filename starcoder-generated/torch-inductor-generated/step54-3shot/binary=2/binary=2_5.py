
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=9, padding=10)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 3.0
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
