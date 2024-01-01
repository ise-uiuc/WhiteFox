
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.sigmoid(v1)
        return v3 * v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
