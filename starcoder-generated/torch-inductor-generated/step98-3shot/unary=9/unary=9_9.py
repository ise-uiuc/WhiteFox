
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v = self.conv(x)
        y = torch.nn.functional.relu6(v+3)
        out = y/6.0
        return out
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
