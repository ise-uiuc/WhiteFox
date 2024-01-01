
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(63, 75, 1, stride=1, padding=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3).permute(0, 1, 4, 2, 3)
        return v4.reshape(36339).permute(1, 0)
negative_slope = 0.5
x = torch.randn(1, 223, 1, 1)
