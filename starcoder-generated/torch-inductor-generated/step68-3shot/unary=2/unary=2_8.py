
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, (1, 1), bias=True)
    def forward(self, x, x2):
        v1 = self.conv_transpose(x)
        v2 = v1 * x2
        return v2
# Inputs to the model
x = torch.randn(5, 2, 1, 1)
x2 = torch.randn(1)
