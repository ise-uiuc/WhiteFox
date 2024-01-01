
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 32, 3, stride=2, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1) + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        v5 = self.conv_transpose2(v4) + 3
        v6 = torch.clamp_min(v5, 0)
        v7 = torch.clamp_max(v6, 6)
        v8 = v7 / 6
        v9 = self.conv_transpose3(v8) + 3
        v10 = torch.clamp_min(v9, 0)
        v11 = torch.clamp_max(v10, 6)
        v12 = v11 / 6
        v13 = self.conv_transpose4(v12) + 3
        v14 = torch.clamp_min(v13, 0)
        v15 = torch.clamp_max(v14, 6)
        v16 = v15 / 6
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
