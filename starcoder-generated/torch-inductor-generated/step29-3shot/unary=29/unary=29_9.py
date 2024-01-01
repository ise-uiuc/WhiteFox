
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.7100000381469727, max_value=2.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(512, 2, 1, stride=1, padding=2)
        self.gelu = torch.nn.GELU()
        self.upsample = torch.nn.Upsample(scale_factor=98.0, mode='nearest')
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 8, 3, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(4, 1024, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.gelu(v1)
        v3 = self.upsample(v2)
        v4 = self.conv_transpose1(v2)
        v5 = torch.mul(v4, v3)
        v6 = self.conv(v5)
        v7 = self.gelu(v6)
        v8 = self.conv_transpose(v7)
        v9 = torch.mul(x1, v8)
        v10 = torch.clamp_min(v9, self.min_value)
        v11 = torch.clamp_max(v10, self.max_value)
        v12 = self.conv_transpose1(v11)
        v13 = torch.mul(v12, v6)
        v14 = torch.clamp_min(v13, self.min_value)
        v15 = torch.clamp_max(v14, self.max_value)
        v16 = self.conv(v7)
        v17 = torch.mul(v15, v16)
        v18 = torch.clamp_min(v17, self.min_value)
        v19 = torch.clamp_max(v18, self.max_value)
        return v19
# Inputs to the model
x1 = torch.randn(1, 512, 8, 8)
