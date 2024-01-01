
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.Conv2d(102, 140, 1, stride=1, padding=0)
    def forward(self, x10):
        v1 = self.conv_t(x10)
        v2 = v1 > 0
        v3 = v1 * 0.7330
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x10 = torch.randn(25, 102, 32, 9)
