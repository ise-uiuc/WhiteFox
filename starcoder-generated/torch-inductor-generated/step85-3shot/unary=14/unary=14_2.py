
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(3, 56, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_8(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = torch.nn.functional.max_pool2d(v3, 3, stride=2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
