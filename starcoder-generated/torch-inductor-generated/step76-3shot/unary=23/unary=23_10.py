
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 10, 3, stride=5, padding=3, dilation=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 5, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 20, 10, 10)
