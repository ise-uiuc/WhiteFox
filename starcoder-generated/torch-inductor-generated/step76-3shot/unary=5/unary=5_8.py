
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(2)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, 6, stride=2, padding=2)
    def forward(self, x1):
        v1 = v6 = self.avg_pool(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 50, 40)
