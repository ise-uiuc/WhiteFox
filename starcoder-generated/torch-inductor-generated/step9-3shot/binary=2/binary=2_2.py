
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v = self.conv(x)
        return v - 1.0249
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
