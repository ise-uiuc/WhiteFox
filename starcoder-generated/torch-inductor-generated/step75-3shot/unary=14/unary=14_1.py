
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(226, 86, 2, stride=2, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 226, 16, 16)
