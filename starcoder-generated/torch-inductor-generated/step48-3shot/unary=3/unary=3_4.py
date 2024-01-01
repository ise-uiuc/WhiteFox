
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 * 0.5
        v5 = v3 + 1.0
        v6 =  v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 59, 69)
