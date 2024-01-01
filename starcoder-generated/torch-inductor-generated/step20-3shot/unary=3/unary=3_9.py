
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 7, stride=7, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 2.0
        v3 = v1 * 2.3284271247461903
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
