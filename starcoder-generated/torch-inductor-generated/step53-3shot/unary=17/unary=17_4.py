
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
