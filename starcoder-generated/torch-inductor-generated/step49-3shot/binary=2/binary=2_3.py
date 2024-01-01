
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 6, 4, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.01
        return v2
# Inputs to the model
x = torch.randn(1, 4, 56, 56)
