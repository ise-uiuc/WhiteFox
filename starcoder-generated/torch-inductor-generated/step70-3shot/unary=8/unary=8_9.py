
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(256, 128, 2, stride=1, padding=2, bias=False)
        self.conv_transpose_ = torch.nn.ConvTranspose2d(64, 128, 3, stride=1, padding=2, bias=False)
        self.conv_transpose__ = torch.nn.ConvTranspose3d(64, 128, 4, stride=1, padding=2, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_(v1)
        v3 = self.conv_transpose__(v2)
        v4 = v1 + 3
        v5 = torch.clamp(v4, min=0)
        v6 = v3 + v5
        return torch.abs(v6)
# Inputs to the model
x1 = torch.randn(2, 256, 64)
