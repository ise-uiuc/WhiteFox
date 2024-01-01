
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1).add(3).clamp(0, 6)[0, :, :, 0][:, :, None, None].div(6)
        v2 = self.conv(v1).mul(0.00390625).div(6)
        return v2.transpose(0, 1).mul(0.00390625).div(6)
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
