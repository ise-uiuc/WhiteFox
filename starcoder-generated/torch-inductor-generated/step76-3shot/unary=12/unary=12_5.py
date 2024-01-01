
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(3, 3, 2, stride=2, padding=0, groups=3)
        self.pointwise = torch.nn.Conv2d(3, 4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.depthwise(x1)
        v2 = self.pointwise(v1)
        v3 = self.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
