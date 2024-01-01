
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 1, 3, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 5, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 7, 9)
