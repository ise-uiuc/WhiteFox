
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=1)
        self.avg_pool2d = torch.nn.AvgPool2d(1, stride=1, padding=1, count_include_pad=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.avg_pool2d(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
