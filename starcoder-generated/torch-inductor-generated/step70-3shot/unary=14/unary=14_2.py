
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_26_2 = torch.nn.ConvTranspose2d(60, 45, 5, stride=1, padding=0, bias=False)
    def forward(self, x1, x2, x3):
        v1 = self.conv_transpose_26_2(x1)
        v2 = x2.matmul(x3)
        v3 = v2 > v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 60, 8, 8)
x2 = torch.randn(1, 8, 8)
x3 = torch.randn(1, 45, 8, 8)
