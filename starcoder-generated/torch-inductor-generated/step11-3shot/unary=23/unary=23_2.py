
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 4, 16, stride=8, padding=8)
    def forward(self, x1):
        v1 = torch.tanh(self.conv_transpose(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
