
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(35, 32, 3, stride=1, padding=2)
        self.conv = torch.nn.ConvTranspose2d(32, 96, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 35, 10, 10)
