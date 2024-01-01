
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 2, 1, stride=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
