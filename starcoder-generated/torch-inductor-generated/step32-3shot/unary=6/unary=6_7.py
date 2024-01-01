
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_point = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.conv_x = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv_point(x1)
        v2 = x2
        v3 = self.conv_x(v2)
        v4 = 3 + v1
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
