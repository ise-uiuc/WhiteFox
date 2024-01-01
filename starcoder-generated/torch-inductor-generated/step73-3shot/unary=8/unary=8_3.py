
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 2, 2, stride=2, dilation=2, padding=2, output_padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 34, 3, padding=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(34, 29, 4, stride=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(29, 5, 2, stride=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(5, 8, 4, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        v8 = self.conv_transpose_1(v7)
        v9 = v8 + 3
        v10 = torch.clamp(v9, min=0)
        v11 = torch.clamp(v10, max=6)
        v12 = v8 * v11
        v13 = v12 / 6
        v14 = self.conv_transpose_2(v13)
        v15 = v14 + 3
        v16 = torch.clamp(v15, min=0)
        v17 = torch.clamp(v16, max=6)
        v18 = v14 * v17
        v19 = v18 / 6
        v20, v21, v22 = self.conv_transpose_3(v19).chunk(3, dim=1)
        output = v20
        return output
# Inputs to the model
x1 = torch.randn(1, 23, 19, 19)
