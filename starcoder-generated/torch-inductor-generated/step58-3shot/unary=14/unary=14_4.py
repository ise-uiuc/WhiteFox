
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(1, 16, 1, stride=1, bias=False, dilation=1)
        self.linear_1 = torch.nn.Linear(6272, 10)
    def forward(self, x1):
        v1 = self.conv_transpose_15(x1)
        v1 = v1.reshape(v1.size(0), -1)
        v2 = self.linear_1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(4, 6272)
