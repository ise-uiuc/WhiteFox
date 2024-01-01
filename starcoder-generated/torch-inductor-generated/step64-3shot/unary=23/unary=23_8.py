
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 3, stride=2)
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
