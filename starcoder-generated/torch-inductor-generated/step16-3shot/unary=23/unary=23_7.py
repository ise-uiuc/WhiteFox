
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, bias=False)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 1, 500)
