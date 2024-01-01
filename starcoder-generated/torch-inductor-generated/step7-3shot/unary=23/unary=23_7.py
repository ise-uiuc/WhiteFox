
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 7, 2, stride=2, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(7, 6, 4, stride=3, padding=2)
        self.conv = torch.nn.ConvTranspose1d(3, 6, 3)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
