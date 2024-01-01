
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
    def forward(self, x):
        v = self.conv(x)
        return v * 1.5
# Inputs to the model
x = torch.randn(64, 64, 64, 64)
