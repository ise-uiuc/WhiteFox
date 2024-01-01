
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 112, 12, stride=1, padding=0, dilation=12)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(self.conv_transpose(x1), size=[128, 128], mode='linear', align_corners=None)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 8, 36, 54)
