
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv = torch.nn.Conv2d(32, 32, 4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 32, 20, 20)
