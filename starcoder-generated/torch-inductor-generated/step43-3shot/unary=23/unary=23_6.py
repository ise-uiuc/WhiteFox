
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(2, 2, 5, stride=2, padding=0)
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 6, 6)
