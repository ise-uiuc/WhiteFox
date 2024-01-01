
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x, y):
        v1 = self.conv(x)
        v2 = v1 - v1
        v3 = v2 - y
        v4 = v2 - y
        return v3, v4
# Inputs to the model
x = torch.randn(1, 1, 10, 10)
y = torch.randn(1, 1, 10, 10)
