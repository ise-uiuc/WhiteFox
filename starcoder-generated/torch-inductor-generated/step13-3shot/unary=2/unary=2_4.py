
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(3, 3, 2, bias=True)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, 2, 1, 2, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(3, 3, 2, 1, stride=(2, 1, 1), groups=1, bias=True)
    def forward(self, x1, x2, x3):
        v1 = self.conv_transpose_0(x1)
        v2 = self.conv_transpose_1(x2)
        v3 = self.conv_transpose_2(x3)
        v4 = v1 + v2
        v5 = v4 + v3
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
x2 = torch.randn(1, 3, 2, 9)
x3 = torch.randn(1, 3, 10, 20)
