
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 1, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv_transpose_1(x3)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = torch.mul(input=v2, other=v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 21, 21)
x2 = torch.randn(1, 3, 22, 22)
x3 = torch.randn(1, 3, 23, 23)
