
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 16, stride=1, padding=3)
    def forward(self, x):
        v = self.conv(x)
        return v
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
