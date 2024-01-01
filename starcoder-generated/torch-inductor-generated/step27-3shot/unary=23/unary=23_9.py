
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 2, 2, stride=2)
    def forward(self, x1):
        v1 = torch.tanh(self.conv_transpose(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 300)
