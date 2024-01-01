
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(11, 1), stride=(3, 2), padding=(1, 3))
        self.conv_transpose = torch.nn.ConvTranspose1d(1, 1, kernel_size=3, stride=3, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
