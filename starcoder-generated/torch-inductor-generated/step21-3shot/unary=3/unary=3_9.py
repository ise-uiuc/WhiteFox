
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(13, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.6931471805601534
        v5 = v3 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 13, 64, 64)
