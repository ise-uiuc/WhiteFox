
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(1, 1, [17, 17], stride=[4, 4], padding=[0, 0], groups=1, bias=False)
        self.pointwise = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.depthwise(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return self.pointwise(v6)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
