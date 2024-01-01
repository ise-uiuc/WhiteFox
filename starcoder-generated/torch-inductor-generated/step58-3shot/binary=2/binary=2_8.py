
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - [1.0, 0.0, -2.0, 0.0, -3.0, 0.0, -1.0, 1.0, -4.0, 1.0, 4.0, 0.0, 4.0, -1.0, -3.0, -3.0, 2.0, -3.0, 0.0, -2.0, 0.0, 2.0, 1.0, 0.0, 1.0, -2.0, 2.0, 3.0, 4.0]
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
