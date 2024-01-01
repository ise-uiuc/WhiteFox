
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x):
        v = self.conv(x)
        s = 3.14159
        return v - s
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
