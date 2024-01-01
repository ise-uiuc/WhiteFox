
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 7, stride=7)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.3
        return v2
# Inputs to the model
x = torch.randn(1, 64, 32, 96)
