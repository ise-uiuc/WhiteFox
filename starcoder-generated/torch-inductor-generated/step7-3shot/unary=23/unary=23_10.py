
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 7, 2, stride=2, padding=1)
        self.conv = torch.nn.ConvTranspose2d(7, 7, 4, stride=3, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
