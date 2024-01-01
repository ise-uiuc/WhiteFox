
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 0.42
        self.conv_t = torch.nn.ConvTranspose2d(1, 5, 9, stride=4)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(4, 1, 1, 1, device='cuda')
