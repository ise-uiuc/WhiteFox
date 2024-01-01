
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x4):
        v1 = self.conv_(x4)
        v2 = v1 > 0
        v3 = v1 * 0.67
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x4 = torch.randn(6, 3, 8, 8)
