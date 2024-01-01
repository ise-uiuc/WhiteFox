
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1,3, 3, stride=1, padding=1, output_padding=1)
    def forward(self, x2):
        v2 = self.conv_transpose(x2)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x2 = torch.randn(1, 1, 8, 64)
