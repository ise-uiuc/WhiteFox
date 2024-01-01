
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = v1 * v2
        return v1, v3
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
x2 = torch.randn(1, 1, 224, 224)
