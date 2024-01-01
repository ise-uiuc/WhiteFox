
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 9, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv(x)
        v3 = torch.add(v1, v2)
        return v3
# Inputs to the model
x = torch.randn(1, 4, 64, 64)
