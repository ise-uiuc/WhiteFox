
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 3, stride=2, padding=1, groups=7, bias=False)
        self.max_pool2d = torch.nn.MaxPool2d(2, stride=1, padding=1)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.max_pool2d(v5)
        v7 = self.gelu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
