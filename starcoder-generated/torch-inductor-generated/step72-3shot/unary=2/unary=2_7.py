
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 5, 9, 3, 0, bias=True)
        self.conv2d2 = torch.nn.Conv2d(5, 6, 6, 5, 0, bias=True)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv2d2(v1)
        v3 = v2 * v2
        v4 = v2 + v3
        v5 = torch.sigmoid(v4)
        v6 = v5 * 5
        return v6
# Inputs to the model
x1 = torch.randn(32, 3, 63, 27)
