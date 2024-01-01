
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 5, 7, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.conv_transpose1d(x1, self.conv_transpose.weight, self.conv_transpose.bias, stride=1, padding=3, output_padding=2)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 10)
