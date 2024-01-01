
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(192, 96, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(96, 192, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 192, 112, 112)
