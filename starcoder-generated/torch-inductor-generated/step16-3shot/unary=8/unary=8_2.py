
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose3 = torch.nn.ConvTranspose2d(90, 8, 1, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 4, 4, stride=4)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(4, 2, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose3(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.conv_transpose2(v6)
        v8 = v7 + 3
        v9 = torch.clamp(v8, min=0)
        v10 = torch.clamp(v9, max=6)
        v11 = v7 * v10
        v12 = v11 / 6
        v13 = self.conv_transpose1(v12)
        v14 = v13 + 3
        v15 = torch.clamp(v14, min=0)
        v16 = torch.clamp(v15, max=6)
        v17 = v13 * v16
        v18 = v17 / 6
        return v18
# Inputs to the model
x1 = torch.randn(1, 90, 62, 13)
