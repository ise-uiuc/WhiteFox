
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1)
        self.activation_layer = torch.nn.GELU(True)
    def forward(self, x6):
        v1 = self.conv_transpose(x6)
        v2 = self.activation_layer(v1)
        return v2
# Inputs to the model
x6 = torch.randn(1, 3, 32, 32)
