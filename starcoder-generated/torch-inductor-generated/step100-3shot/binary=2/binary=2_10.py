
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 45.602
        return v2
# Inputs to the model
x = torch.randn(1, 10, 64, 64)
