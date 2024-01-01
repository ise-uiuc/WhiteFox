
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1)
        self.softmax = torch.nn.Softmax2d()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.softmax(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
