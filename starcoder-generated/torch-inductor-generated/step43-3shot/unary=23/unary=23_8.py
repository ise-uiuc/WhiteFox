
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2, kernel_size=3, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 3, 127, 127)
