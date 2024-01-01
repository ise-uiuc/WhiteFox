
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_14 = torch.nn.ConvTranspose2d(2, 1, 7, stride=2, padding=1)
    def forward(self, x1):
        v14 = self.conv_transpose_14(x1)
        return v14
# Inputs to the model
x1 = torch.randn(1, 2, 39, 39)
