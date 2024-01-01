
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, padding=2, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.LeakyReLU(0.1)(v1)
        return torch.nn.Sigmoid()(v2)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
