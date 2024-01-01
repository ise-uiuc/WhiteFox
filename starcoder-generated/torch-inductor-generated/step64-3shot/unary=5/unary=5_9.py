
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose3d(2, 10, 5, stride=2, padding=4, output_padding=4)
        self.conv_transpose2 = torch.nn.ConvTranspose3d(10, 10, 11, stride=11, padding=4)
        self.conv_transpose3 = torch.nn.ConvTranspose3d(10, 4, 1, stride=13, padding=0, output_padding=0)
        self.conv_transpose4 = torch.nn.ConvTranspose3d(4, 7, 2, stride=4, padding=0)
        self.conv_transpose5 = torch.nn.ConvTranspose3d(10, 4, 1, stride=13, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv_transpose2(v6)
        v8 = self.conv_transpose3(v7)
        v9 = self.conv_transpose4(v6)
        v10 = v9 * 0.5
        v11 = v9 * 0.7071067811865476
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        v15 = self.conv_transpose5(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 2, 11, 11, 6)
