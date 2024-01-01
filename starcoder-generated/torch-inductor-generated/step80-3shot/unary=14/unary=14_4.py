
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(4, 1, 1, stride=1, padding=0)
        self.sigmoid_1 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_transpose_8(x1)
        v2 = self.sigmoid_1(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 12, 12)
