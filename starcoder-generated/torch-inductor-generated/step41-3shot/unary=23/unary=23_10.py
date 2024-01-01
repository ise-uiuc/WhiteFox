
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(4, 5, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tan(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 8)
