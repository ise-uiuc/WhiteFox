
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(27,15,7)
    def forward(self, x2):
        v2 = self.conv_transpose(x2)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x2 = torch.randn(1, 27, 19, 127, dtype=torch.float32)
