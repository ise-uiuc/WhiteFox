
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(2332, 73, 5, stride=2, padding=2, dilation=1, groups=4)
        self.relu_1 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose_11(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v7 = self.relu_1(v3)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2332, 2, 2)
