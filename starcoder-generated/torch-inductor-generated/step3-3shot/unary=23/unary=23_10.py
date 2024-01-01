
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(2, 3, 3, 2, 1)
    def forward(self, x1):
        v1 = self.conv_transpose_3(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 9, 9)
