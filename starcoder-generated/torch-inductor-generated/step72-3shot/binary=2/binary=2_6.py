
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 11, stride=11, padding=12)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = (v1 * 5.5) + x * 0.5
        return v2 - 3.5
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
