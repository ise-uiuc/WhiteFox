
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose_conv_2 = torch.nn.ConvTranspose2d(17, 17, 1, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.transpose_conv_2(x1)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 17, 64, 64)
