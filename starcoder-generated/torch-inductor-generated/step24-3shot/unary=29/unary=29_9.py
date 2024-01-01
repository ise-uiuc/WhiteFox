
class Model(torch.nn.Module):
    def __init__(self, min_value=max_value, max_value=0):
        super().__init__()
        for i in range(4):
            setattr(self, "conv_transpose_"+str(i), torch.nn.ConvTranspose2d(8, 3, 1, stride=1, padding=1))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv_transpose_0(x1)
        v2 = self.conv_transpose_1(x2)
        v3 = self.conv_transpose_2(x3)
        v4 = self.conv_transpose_3(x4)
        v5 = torch.clamp_min(v1, self.min_value)
        v6 = torch.clamp_min(v2, self.min_value)
        v7 = torch.clamp_min(v3, self.min_value)
        v8 = torch.clamp_min(v4, self.min_value)
        v9 = torch.clamp_max(v5, self.min_value)
        v10 = torch.clamp_max(v6, self.min_value)
        v11 = torch.clamp_max(v7, self.min_value)
        v12 = torch.clamp_max(v8, self.min_value)
        v13 = torch.clamp_max(v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12, self.max_value)
        v14 = torch.clamp_max(v9, self.max_value)
        v15 = torch.clamp_max(v10, self.max_value)
        v16 = torch.clamp_max(v11, self.max_value)
        v17 = torch.clamp_max(v12, self.max_value)
        v18 = torch.clamp_max(v13 + v14 + v15 + v16 + v17, self.max_value)
        return v18
# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)
x2 = torch.randn(1, 8, 1, 1)
x3 = torch.randn(1, 8, 128, 128)
x4 = torch.randn(1, 8, 256, 256)
