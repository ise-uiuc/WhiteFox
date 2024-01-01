
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 2, 3, stride=1, padding=1, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.sigmoid(self.conv(x1))
        v2 = v1 * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 20, 64, 64)
