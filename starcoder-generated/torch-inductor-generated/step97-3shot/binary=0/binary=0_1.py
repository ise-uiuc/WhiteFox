
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        padding, feature = x1, x2
        v1 = self.conv(feature)
        if padding == x3:
            padding = torch.randn(v1.shape)
        v2 = v1 + padding
        feature = x4
        v3 = v2 + feature
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
x3 = x1
x4 = x2
