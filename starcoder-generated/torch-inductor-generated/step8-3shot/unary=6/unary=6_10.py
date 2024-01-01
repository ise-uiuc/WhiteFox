
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pos = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv_neg = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv_scale = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        x3 = torch.relu(self.conv_pos(x1))
        x4 = torch.sigmoid(self.conv_neg(x1))
        v2 = self.conv_scale(x1)
        v3 = x3 + v2
        v4 = torch.clamp(v3, 0, 6)
        v5 = v4 * v2
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
