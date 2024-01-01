
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 15, 4, stride=2, padding=3, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(15, 14, 5, stride=3, padding=6, output_padding=7)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(14, 15, 5, stride=2, padding=4, output_padding=0)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(15, 16, 5, stride=2, padding=6, output_padding=6)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(v3)
        v5 = v1 + 3
        v6 = torch.clamp(v5, min=0)
        v7 = torch.clamp(v6, max=6)
        v8 = v2 + 3
        v9 = torch.clamp(v8, min=0)
        v10 = torch.clamp(v9, max=6)
        v11 = v3 + 3
        v12 = torch.clamp(v11, min=0)
        v13 = torch.clamp(v12, max=6)
        v14 = v4 + 3
        v15 = torch.clamp(v14, min=0)
        v16 = torch.clamp(v15, max=6)
        v17 = v5 * v7
        v18 = v8 * v10
        v19 = v11 * v13
        v20 = v14 * v16
        v21 = v17 + v20
        v22 = v18 + v19
        v23 = v21 + v22
        return v23
# Inputs to the model
x1 = torch.randn(1, 1, 15, 15)
