
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(6, 16, 5, stride=2, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.relu(v1)
        v3 = torch.mul(x1, v1)
        return v3
# Inputs to the model
x1 = torch.randn(3, 15, 15, 32)
