
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.layer = torch.nn.Sequential(self.conv)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.layer(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 16, 56, 56)
