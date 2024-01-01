
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 33, 2, stride=2, padding=1)
    def forward(self, x1):
        conv_transpose = self.conv_transpose(x1)
        sigmoid = torch.sigmoid(conv_transpose)
        multiply = conv_transpose * sigmoid
        return multiply
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
