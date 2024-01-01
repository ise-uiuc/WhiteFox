
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Conv2d(3, 1, 2, 1, 3)
        self.conv = torch.nn.ConvTranspose2d(3, 3, 3, 1, 0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 35, 35)
