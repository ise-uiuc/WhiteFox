
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 56, 1, stride=(1, 2), padding=0)
        self.conv = torch.nn.Conv2d(6, 40, 1, stride=(1, 2), padding=0)
    def forward(self, x3):
        v1 = self.conv_transpose(x3)
        v2 = self.conv(x3)
        v3 = v1 > 0
        v4 = v2 > 0
        v5 = torch.logical_and(v3, v2)
        v6 = torch.logical_and(v4, v5)
        v7 = v2 * 2.727
        v8 = torch.where(v6, v1, v7)
        return v8
# Inputs to the model
x3 = torch.randn(1, 6, 16, 8)
