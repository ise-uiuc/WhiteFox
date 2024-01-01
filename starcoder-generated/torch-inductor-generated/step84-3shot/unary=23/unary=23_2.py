
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 5, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 8, 5, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv_transpose(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
