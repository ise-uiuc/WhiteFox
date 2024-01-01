
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_op = torch.nn.ConvTranspose2d(3, 2, 4, stride=2, out_padding=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_op(x1)
        v2 = v1 - 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 25, 25)
