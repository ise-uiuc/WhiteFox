
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 5, stride=1, padding=2)
    def forward(self, x1):
        v2 = torch.tanh(self.conv_transpose(x1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 13, 13)
