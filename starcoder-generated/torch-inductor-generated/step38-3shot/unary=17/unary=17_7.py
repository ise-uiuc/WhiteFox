
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 7, stride=2)
    def forward(self, x1):
        v1 = torch.tanh(self.conv(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 192, 100)
