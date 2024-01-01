
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.conv_transpose3d(x1, torch.randn(21, 6, 5, 5, 3), stride=2, groups=1, padding=4, dilation=1, out_padding=1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 9, 16, 16, 16)
