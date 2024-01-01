
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 8, 1, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x = torch.randn(1, 2, 35, 35)
